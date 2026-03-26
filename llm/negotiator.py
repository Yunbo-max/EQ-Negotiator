"""
Multi-Agent Credit Negotiation System for EQ-Negotiator.
Based on Algorithm 1 and Section 3.3 of the paper.

Three agents: Creditor (M_creditor), Debtor (M_debtor), Judge (M_judge)
"""

import re
import json
from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage
from llm.llm_wrapper import LLMWrapper
from models.eq_negotiator import EQNegotiator
from models.constants import EMOTIONS, EMOTION_PROMPTS


class GameState(Dict):
    messages: List
    turn: str
    product: Dict
    seller_config: Dict
    buyer_config: Dict
    history: List
    current_state: str


class EQDebtNegotiator:
    """
    Multi-agent debt negotiation with EQ-Negotiator emotion engine.
    Implements Algorithm 1 from the paper.
    """

    def __init__(self, config: Dict[str, Any],
                 eq_engine: EQNegotiator,
                 model_creditor: str = "gpt-4o-mini",
                 model_debtor: str = "gpt-4o-mini",
                 debtor_persona: str = "vanilla"):
        self.config = config
        self.eq_engine = eq_engine
        self.debtor_persona = debtor_persona

        self.llm_creditor = LLMWrapper(model_creditor, "creditor")
        self.llm_debtor = LLMWrapper(model_debtor, "debtor")

        self.negotiation_round = 0
        self.emotion_sequence = []

    def detect_emotion(self, message: str) -> str:
        """
        In-context emotion recognition (Section 3.1).
        Uses LLM to classify debtor emotion into 7 categories.
        """
        if not message:
            return "Neutral"

        prompt = f"""Analyze the emotional tone of this debt negotiation message.
Classify into exactly ONE of: Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral

CONTEXT RULES for debt collection:
- Resistance, defensiveness, or pushback -> Anger
- Pleading, financial hardship -> Sadness
- Threats, intimidation -> Anger
- Deception, manipulation -> Disgust
- Anxiety about consequences -> Fear
- Cooperative, agreeable -> Joy
- Factual, business-like -> Neutral

MESSAGE: "{message}"

Respond with ONLY the emotion word:"""

        try:
            response = self.llm_creditor.invoke([HumanMessage(content=prompt)])
            detected = response.content.strip()
            # Validate
            for e in EMOTIONS:
                if e.lower() == detected.lower():
                    return e
            return "Neutral"
        except Exception:
            return "Neutral"

    def extract_days(self, text: str) -> Optional[int]:
        """Extract payment timeline in days from text."""
        if not text:
            return None
        patterns = [
            (r'(\d+)\s*days?', 1),
            (r'(\d+)\s*weeks?', 7),
            (r'(\d+)\s*months?', 30),
        ]
        for pattern, multiplier in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                return int(matches[-1]) * multiplier
        return None

    def judge_agreement(self, creditor_msg: str, debtor_msg: str,
                        creditor_days: int, debtor_days: int) -> Dict:
        """Judge agent: evaluate if agreement reached (Section 3.3)."""
        prompt = f"""You are an impartial JUDGE analyzing a debt negotiation.

Creditor's message: "{creditor_msg}"
Debtor's message: "{debtor_msg}"
Creditor proposed: {creditor_days} days
Debtor proposed: {debtor_days} days
Difference: {abs(creditor_days - debtor_days) if creditor_days and debtor_days else 'Unknown'} days

RULES:
- Timeline difference <= 5 days without rejection = AGREEMENT
- Explicit acceptance ("I accept", "deal", "agreed") = AGREEMENT

Respond with ONLY JSON:
{{"agreement_reached": true/false, "final_days": number_or_null, "reasoning": "brief"}}"""

        try:
            response = self.llm_creditor.invoke([HumanMessage(content=prompt)])
            match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if match:
                result = json.loads(match.group())
                return {"agreement": result.get("agreement_reached", False),
                        "final_days": result.get("final_days"),
                        "reasoning": result.get("reasoning", "")}
        except Exception:
            pass
        return {"agreement": False, "reasoning": "Judge error"}

    def creditor_node(self, state: GameState):
        """Creditor agent with EQ-Negotiator emotion engine."""
        self.negotiation_round += 1

        # Detect debtor emotion from last message
        debtor_emotion = "Neutral"
        history = state.get("history", [])
        for speaker, msg in reversed(history):
            if speaker == "buyer":
                debtor_emotion = self.detect_emotion(msg)
                break

        # EQ-Negotiator emotion selection (Eq. 4)
        emotion_config = self.eq_engine.select_emotion(debtor_emotion)
        creditor_emotion = emotion_config['emotion']
        self.emotion_sequence.append(creditor_emotion)

        strategy = emotion_config['strategy']
        print(f"        [{strategy}] Debtor: {debtor_emotion} -> Creditor: {creditor_emotion} "
              f"(State: {emotion_config['dominant_state']})")

        # Build creditor prompt
        config = self.config.get("seller_config", self.config["seller"])
        debt_info = self.config.get('metadata', {})
        balance = debt_info.get('outstanding_balance', 0)

        # Timeline tracking
        creditor_days_list, debtor_days_list = [], []
        for speaker, msg in history:
            days = self.extract_days(msg)
            if days:
                if speaker == "seller":
                    creditor_days_list.append(days)
                elif speaker == "buyer":
                    debtor_days_list.append(days)

        timeline_text = ""
        if creditor_days_list:
            timeline_text += f"Your previous offer: {creditor_days_list[-1]} days. "
        if debtor_days_list:
            timeline_text += f"Debtor requested: {debtor_days_list[-1]} days. "
            if creditor_days_list:
                gap = abs(debtor_days_list[-1] - creditor_days_list[-1])
                if gap <= 10:
                    timeline_text += f"Gap is small ({gap} days) - consider accepting."
                else:
                    mid = (creditor_days_list[-1] + debtor_days_list[-1]) // 2
                    timeline_text += f"Consider moving toward {mid} days."

        prompt = f"""You are a Creditor negotiating payment timeline with the Debtor.

ROLE: Speak ONLY as the Creditor. No labels. 1-2 sentences max.

DEBT CONTEXT:
- Outstanding Balance: ${balance:,.2f}
- Your Target: {config['target_price']} days
- Recovery Stage: {debt_info.get('recovery_stage', 'Collection')}

{timeline_text}

EMOTIONAL APPROACH:
{emotion_config['emotion_text']}

RULES:
- Make gradual concessions toward debtor's position
- When gap is within 5-10 days, consider accepting
- Never copy debtor's exact number immediately

Respond with your counter-offer:"""

        response = self.llm_creditor.invoke(
            [HumanMessage(content=prompt)],
            temperature=emotion_config.get('temperature', 0.7)
        )

        new_history = state["history"] + [("seller", response.content)]

        # Check convergence
        current_state = "offer"
        seller_days = self.extract_days(response.content)
        buyer_days = None
        for speaker, msg in reversed(new_history[:-1]):
            if speaker == "buyer":
                buyer_days = self.extract_days(msg)
                if buyer_days:
                    break

        if seller_days and buyer_days and abs(seller_days - buyer_days) <= 5:
            current_state = "accept"

        return {
            "messages": [response], "turn": "buyer",
            "current_state": current_state, "history": new_history,
        }

    def debtor_node(self, state: GameState):
        """Debtor agent with configurable persona."""
        config = self.config.get("buyer_config", self.config["buyer"])
        debt_info = self.config.get('metadata', {})
        balance = debt_info.get('outstanding_balance', 0)

        # Persona-specific prompt additions
        persona_prompt = ""
        if self.debtor_persona == "angry":
            persona_prompt = "\nMaintain an angry, frustrated tone throughout."
        elif self.debtor_persona == "sad":
            persona_prompt = "\nShow distress about your financial situation."
        elif self.debtor_persona == "threatening":
            persona_prompt = "\nUse intimidation tactics - threaten legal action, complaints."
        elif self.debtor_persona == "cheating":
            persona_prompt = "\nUse deceptive tactics - exaggerate hardship, make false promises."
        elif self.debtor_persona == "victim":
            persona_prompt = "\nPlay the victim - guilt-trip, emphasize unfairness."
        elif self.debtor_persona == "stonewalling":
            persona_prompt = "\nBe deliberately unresponsive, give minimal answers, delay."
        elif self.debtor_persona == "fear":
            persona_prompt = "\nShow anxiety and nervousness about consequences."
        elif self.debtor_persona == "disgust":
            persona_prompt = "\nExpress contempt and disappointment about the situation."

        prompt = f"""You are the Debtor negotiating payment timeline with the Creditor.

ROLE: Speak ONLY as the Debtor. No labels. 1-2 sentences max.
{persona_prompt}

YOUR SITUATION:
- Outstanding Balance: ${balance:,.2f}
- Cash Flow: {debt_info.get('cash_flow_situation', 'Tight')}
- Your Target: {config['target_price']} days

RULES:
- Make gradual concessions toward creditor's position
- When gap is within 5-10 days, consider accepting

Respond:"""

        response = self.llm_debtor.invoke([HumanMessage(content=prompt)])
        new_history = state["history"] + [("buyer", response.content)]

        # Check convergence
        current_state = "offer"
        buyer_days = self.extract_days(response.content)
        seller_days = None
        for speaker, msg in reversed(new_history[:-1]):
            if speaker == "seller":
                seller_days = self.extract_days(msg)
                if seller_days:
                    break

        if buyer_days and seller_days and abs(buyer_days - seller_days) <= 5:
            current_state = "accept"

        return {
            "messages": [response], "turn": "seller",
            "current_state": current_state, "history": new_history,
        }

    def should_continue(self, state: GameState):
        if state["current_state"] in ["accept", "breakdown"]:
            return "end"

        if len(state["history"]) >= 2:
            last_two = state["history"][-2:]
            d1 = self.extract_days(last_two[0][1])
            d2 = self.extract_days(last_two[1][1])
            if d1 and d2 and abs(d1 - d2) <= 5:
                return "end"

        return state.get("turn", "buyer")

    def run_negotiation(self, max_dialog_len: int = 30) -> Dict[str, Any]:
        """Run a complete negotiation (Algorithm 1)."""
        workflow = StateGraph(GameState)
        workflow.add_node("seller", self.creditor_node)
        workflow.add_node("buyer", self.debtor_node)
        workflow.add_edge(START, "seller")
        workflow.add_conditional_edges("seller", self.should_continue, {"buyer": "buyer", "end": END})
        workflow.add_conditional_edges("buyer", self.should_continue, {"seller": "seller", "end": END})
        app = workflow.compile()

        debt_info = self.config.get('metadata', {})
        balance = debt_info.get('outstanding_balance', 0)
        target = self.config['seller']['target_price']

        initial_state = GameState(
            messages=[HumanMessage(content=f"Discuss ${balance:,.2f} balance, proposing {target} days.")],
            turn="seller", product=self.config["product"],
            seller_config=self.config["seller"], buyer_config=self.config["buyer"],
            history=[], current_state="offer"
        )

        dialog = []
        final_state = "breakdown"

        for i, step in enumerate(app.stream(initial_state, {"recursion_limit": max_dialog_len * 2})):
            if i > max_dialog_len:
                break
            for node, value in step.items():
                msg = value["messages"][-1].content
                days = self.extract_days(msg)
                dialog.append({"turn": i+1, "speaker": node, "message": msg,
                               "state": value["current_state"], "requested_days": days})
                if value["current_state"] in ["accept", "breakdown"]:
                    final_state = value["current_state"]
                    break

        # Judge final agreement
        if len(dialog) >= 2:
            last_c, last_d = None, None
            last_c_days, last_d_days = None, None
            for entry in reversed(dialog):
                if entry["speaker"] == "seller" and last_c is None:
                    last_c, last_c_days = entry["message"], entry["requested_days"]
                elif entry["speaker"] == "buyer" and last_d is None:
                    last_d, last_d_days = entry["message"], entry["requested_days"]
                if last_c and last_d:
                    break

            if last_c and last_d:
                judge = self.judge_agreement(last_c, last_d, last_c_days, last_d_days)
                if judge["agreement"]:
                    final_state = "accept"

        # Extract final days
        final_days = None
        if final_state == "accept":
            for entry in reversed(dialog):
                if entry["requested_days"]:
                    final_days = entry["requested_days"]
                    break

        # Update EQ engine
        result = {
            "scenario_id": self.config['id'],
            "final_state": final_state,
            "collection_days": final_days,
            "creditor_target_days": int(self.config['seller']['target_price']),
            "negotiation_rounds": len(dialog),
            "emotion_sequence": self.emotion_sequence.copy(),
            "debtor_persona": self.debtor_persona,
            "dialog": dialog,
            "eq_stats": self.eq_engine.get_stats(),
        }

        self.eq_engine.update_after_negotiation(result)

        # Reset for next negotiation
        self.negotiation_round = 0
        self.emotion_sequence = []
        self.eq_engine.reset()

        return result

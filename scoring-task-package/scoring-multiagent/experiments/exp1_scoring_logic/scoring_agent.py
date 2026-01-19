import time
import json
import operator
from typing import Annotated, List, TypedDict, Dict, Any

# --- LIBRARY ---
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# ==========================================
# 1. DEFINISI STRUKTUR OUTPUT (Pydantic)
# ==========================================
class ScoringJSON(BaseModel):
    score: str = Field(description="Nilai 0-100")
    correct: str = Field(description="'true' atau 'false'")
    summary: str = Field(description="Penjelasan singkat")
    Misconceptions: str = Field(description="Daftar miskonsepsi")

# ==========================================
# 2. STATE MANAGEMENT
# ==========================================
class AgentState(TypedDict):
    problem: str
    solution: str
    student_code: str
    rubric: str
    messages: Annotated[List[BaseMessage], operator.add]
    iteration_count: int
    final_json: Dict[str, Any]

# ==========================================
# 3. KELAS UTAMA (SESUAI STANDAR INTERFACE)
# ==========================================
class ScoringMultiAgent:
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inisialisasi komponen sesuai standar.
        Config bisa berisi parameter model, temperature, dll.
        """
        self.config = config or {}
        model_name = self.config.get("model", "llama3.2")
        
        # Inisialisasi LLM
        self.llm = ChatOllama(model=model_name, temperature=0, format="json")
        
        # Build Graph sekalian saat init
        self.app = self._build_graph()
        
        # Context Miskonsepsi
        self.misconceptions_list = """
        1. Intentional Bug (IB)
        2. While Demon (WD)
        3. WhileIf / IfWhile
        4. Executed Once (EO)
        5. Drop Through Error (DT)
        6. Infinite Loop
        7. Wrong Order
        """

    def _build_graph(self):
        """Internal function untuk membangun LangGraph"""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("Supervisor", self._supervisor_node)
        workflow.add_node("StyleChecker", self._style_checker_agent)
        workflow.add_node("LogicChecker", self._logic_checker_agent)
        
        workflow.set_entry_point("Supervisor")
        
        workflow.add_conditional_edges("Supervisor", self._router, 
                                     {"run_workers": "StyleChecker", END: END})
        workflow.add_edge("StyleChecker", "LogicChecker")
        workflow.add_edge("LogicChecker", "Supervisor")
        
        return workflow.compile()

    # --- NODE FUNCTIONS (Internal) ---
    
    def _style_checker_agent(self, state: AgentState):
        print("   --> [Style Checker] Working...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a Style Checker. Analyze readability."),
            ("user", "Code:\n{student_code}\n\nReturn JSON with key 'style_analysis'.")
        ])
        chain = prompt | self.llm 
        res = chain.invoke({"student_code": state["student_code"]})
        return {"messages": [HumanMessage(content=f"Style: {res.content}")]}

    def _logic_checker_agent(self, state: AgentState):
        print("   --> [Logic Checker] Working...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a Logic Checker. Check misconceptions:\n{self.misconceptions_list}"),
            ("user", "Problem: {problem}\nSolution: {solution}\nStudent: {student_code}\n\nReturn JSON with key 'logic_analysis'.")
        ])
        chain = prompt | self.llm
        res = chain.invoke({
            "problem": state["problem"], "solution": state["solution"], 
            "student_code": state["student_code"]
        })
        return {"messages": [HumanMessage(content=f"Logic: {res.content}")]}

    def _supervisor_node(self, state: AgentState):
        current_iter = state.get("iteration_count", 0)
        print(f"--- Supervisor: Iterasi {current_iter} ---")
        
        # Limit iterasi (Default 1 agar cepat, bisa diset via config)
        max_iter = self.config.get("max_iterations", 1)
        
        if current_iter >= max_iter:
            print("   --> [Supervisor] Finalizing...")
            final_prompt = f"""
            Generate FINAL JSON.
            Format: {{"score": "0-100", "correct": "true/false", "summary": "text", "Misconceptions": "text"}}
            Rubric: {state['rubric']}
            Reports: {state['messages']}
            """
            try:
                res = self.llm.invoke(final_prompt)
                clean_json = res.content.replace("```json", "").replace("```", "").strip()
                final_dict = json.loads(clean_json)
            except:
                final_dict = {"error": "JSON Parse Fail", "raw": res.content}

            return {"final_json": final_dict, "iteration_count": current_iter + 1}
        
        return {"iteration_count": current_iter + 1}

    def _router(self, state: AgentState):
        if state.get("final_json"): return END
        return "run_workers"

    # --- PUBLIC INTERFACE (Sesuai Standar) ---

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Menjalankan pipeline sesuai standar antarmuka komponen.
        
        Args:
            input_data (dict): Harus berisi 'pseudocode', 'problem', 'solution', 'rubric'
        
        Returns:
            dict: { 'time': float, 'text': str, 'data': dict }
        """
        print(f"🚀 Running Scoring Agent (Model: {self.config.get('model', 'llama3.2')})...")
        start_time = time.time()
        
        initial_state = {
            "problem": input_data.get("problem", ""),
            "solution": input_data.get("solution", ""),
            "student_code": input_data.get("pseudocode", ""),
            "rubric": input_data.get("rubric", ""),
            "messages": [],
            "iteration_count": 1
        }
        
        try:
            output = self.app.invoke(initial_state)
            final_json = output.get("final_json", {})
            
            # Format Text Utama (Summary + Score)
            summary_text = f"Score: {final_json.get('score')}. {final_json.get('summary')}"
            
        except Exception as e:
            print(f"Error: {e}")
            final_json = {"error": str(e)}
            summary_text = "Error during execution."

        end_time = time.time()
        inference_time = end_time - start_time

        # RETURN SESUAI FORMAT STANDAR YANG DIMINTA
        return {
            'time': inference_time,  # Waktu inferensi
            'text': summary_text,    # Hasil utama (string)
            'data': final_json       # Data lengkap (JSON struct)
        }

# ==========================================
# CONTOH PENGGUNAAN (MAIN BLOCK)
# ==========================================
if __name__ == "__main__":
    # 1. Setup Config
    config = {"model": "llama3.2", "max_iterations": 1}
    
    # 2. Inisialisasi Komponen
    scorer = ScoringMultiAgent(config)
    
    # 3. Siapkan Input Data
    inputs = {
        "pseudocode": "x=1; WHILE x<5 DO; PRINT x; ENDWHILE", # Infinite Loop
        "problem": "Cetak angka 1-5",
        "solution": "i=1; WHILE i<=5 DO; PRINT i; i=i+1; ENDWHILE",
        "rubric": "Benar=100"
    }
    
    # 4. Run Component
    result = scorer.run(inputs)
    
    print("\n" + "="*40)
    print("✅ STANDARD OUTPUT RESULT:")
    print("="*40)
    print(json.dumps(result, indent=2))
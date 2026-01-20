"""
task_router.py

Classifies the user's core "intent" to route to the correct specialist.
This is the "Chief" - it decides *what* action to perform.
It does NOT decide "where" to get data (that's retrieval.router).
"""
from __future__ import annotations
from enum import Enum, auto
import re

class TaskIntent(Enum):
    """The set of all possible tasks the AI can perform."""
    # Core Text Manipulation
    PROOFREAD = auto()
    SUMMARIZE = auto()
    ELABORATE = auto()
    CHANGE_TONE = auto()
    TRANSLATE = auto()
    
    # Content Generation & Analysis
    ANALYZE = auto()
    DRAFT_EMAIL = auto()
    BRAINSTORM_IDEAS = auto()

    # SR&ED / Technical Writing
    STRUCTURE_TECHNICAL_REPORT = auto()
    IDENTIFY_TECHNOLOGICAL_UNCERTAINTIES = auto()
    WRITE_TECHNICAL_JUSTIFICATION = auto()
    EXTRACT_TECHNICAL_DETAILS = auto()
    IMPROVE_TECHNICAL_CLARITY = auto()
    GENERATE_EXPERIMENT_LOGS = auto()
    COMPARE_TECHNICAL_APPROACHES = auto()
    GENERATE_TIMELINE = auto()
    STRUCTURE_SRED_242_SECTION = auto()


    # Technical Generation
    GENERATE_CODE = auto()
    GENERATE_EXCEL_FORMULA = auto() 
    GENERATE_SQL_QUERY = auto() 
    GENERATE_REGEX = auto()         
    SOLVE_MATH = auto()
    EXTRACT_INFO = auto()

    # NEW — Manual Client Grading
    GRADE_CLIENT = auto() 

    # Default
    SEARCH_QUERY = auto()   # The default task

INTENT_TRIGGERS = {
    # Most specific commands come first
    
    # Core Text Manipulation
    TaskIntent.PROOFREAD: [r"\bproofread\b", r"\bfix grammar\b", r"\bcorrect this\b", r"\brefine this\b", r"\bmore refined\b", r"\bmore concise\b", r"\bimprove this\b",],
    TaskIntent.SUMMARIZE: [r"\bsummarize\b", r"\breduce this\b", r"\btl;dr\b", r"\bkey points\b"],
    TaskIntent.ELABORATE: [r"\bmake this longer\b", r"\belaborate on\b", r"\bexpand this\b", r"\badd more detail\b"],
    TaskIntent.CHANGE_TONE: [r"\bmake it more professional\b", r"\bmake this casual\b", r"\bchange the tone\b", r"\bmake this more friendly\b"],
    TaskIntent.TRANSLATE: [r"\btranslate this\b", r"\bin french\b", r"\bin spanish\b", r"\bto german\b"],

    # Content Generation & Analysis
    TaskIntent.DRAFT_EMAIL: [r"\bdraft an email\b", r"\bwrite an email\b", r"\bemail about\b"],
    TaskIntent.BRAINSTORM_IDEAS: [r"\bbrainstorm\b", r"\bgive me ideas\b", r"\bideas for\b", r"\bgenerate a list of\b"],
    TaskIntent.ANALYZE: [r"\banalyze this\b", r"\bexecutive summary\b", r"\bmain ideas\b", r"\bwhat are the themes\b"],
    TaskIntent.EXTRACT_INFO: [r"\bextract the names\b", r"\bpull all the dates\b", r"\bget all emails\b", r"\bextract the addresses\b"],

    # Technical Generation
    TaskIntent.GENERATE_CODE: [r"\bwrite a python script\b", r"\bcode example\b", r"\bhow do I code\b", r"\bgenerate code\b"], # <-- BUG FIXED (was TaskVertical)
    TaskIntent.GENERATE_EXCEL_FORMULA: [r"\bexcel formula\b", r"\bgoogle sheets formula\b", r"\bsum cells\b", r"\bvlookup for\b"],
    TaskIntent.GENERATE_SQL_QUERY: [r"\bsql query\b", r"\bselect from\b", r"\bdatabase query\b", r"\bjoin on\b"],
    TaskIntent.GENERATE_REGEX: [r"\bregex\b", r"\bregular expression\b", r"\bmatch this pattern\b", r"\bparse this with regex\b"],

    # NEW — Manual Client Grading
    TaskIntent.GRADE_CLIENT: [r"\bgrade this client\b", r"\brate this client\b", r"\bscore this client\b", r"\bclient evaluation\b", r"\bassign a client grade\b", r"\bclient scoring\b"],
    TaskIntent.SOLVE_MATH: [r"\bcalculate\b", r"\bsolve for x\b", r"\bwhat is\s*\d+", r"\bmath problem\b", r"\bwhat is the integral\b"],

    #SR&ED/TW
    TaskIntent.STRUCTURE_TECHNICAL_REPORT: [r"\bstructure this\b", r"\bturn this into a report\b", r"\bformat this for sr&ed\b", r"\bformat this for a grant\b", r"\bconvert this into sections\b"],
    TaskIntent.IDENTIFY_TECHNOLOGICAL_UNCERTAINTIES: [r"\bidentify technological uncertainties\b", r"\bfind the technological uncertainties\b", r"\bwhat were the technological uncertainties\b", r"\btech uncertainties\b", r"\bwhich parts are technological uncertainties\b"],
    TaskIntent.WRITE_TECHNICAL_JUSTIFICATION: [r"\btechnical justification\b", r"\bjustify this\b", r"\bexplain why this was hard\b", r"\brationale for\b", r"\bwhy this approach\b"],
    TaskIntent.EXTRACT_TECHNICAL_DETAILS: [r"\bextract the technical details\b", r"\bextract experiments\b", r"\bextract challenges\b", r"\bpull engineering details\b", r"\bwhat were the steps\b"],
    TaskIntent.IMPROVE_TECHNICAL_CLARITY: [r"\bmake this clearer\b", r"\bclarify this\b", r"\bmake this more technical\b", r"\brewrite for engineers\b", r"\bclear technical writing\b"],
    TaskIntent.GENERATE_EXPERIMENT_LOGS: [r"\bcreate experiment logs\b", r"\bmake an experiment log\b", r"\bdocument the attempts\b", r"\bcreate a testing log\b", r"\bengineering log\b"],
    TaskIntent.COMPARE_TECHNICAL_APPROACHES: [r"\bcompare approaches\b", r"\bcompare these methods\b", r"\btradeoffs of\b", r"\bcontrast these options\b"],
    TaskIntent.GENERATE_TIMELINE: [r"\bgenerate a timeline\b", r"\bcreate a timeline\b", r"\btimeline of events\b", r"\bmake a project timeline\b", r"\bchronological order\b", r"\bput this in timeline form\b"],
    TaskIntent.STRUCTURE_SRED_242_SECTION: [r"\bdraft the 242\b", r"\bline 242\b", r"\bsred 242\b", r"\bprepare the 242 section\b", r"\bformat this for 242\b", r"\bwrite the 242 section\b"],
}


def classify_intent(user_input: str) -> TaskIntent:
    """
    Classifies the user's input to route to the correct specialist.
    """
    lowered_input = user_input.lower().strip()

    # Check for specific task commands first
    for intent, triggers in INTENT_TRIGGERS.items():
        if any(re.search(trigger, lowered_input) for trigger in triggers):
            return intent
    return TaskIntent.SEARCH_QUERY

def strip_command_from_text(user_input: str) -> str:
    """
    A helper function to remove the command part, leaving just the text.
    
    Example: "proofread this: The management team are..."
    Returns: "The management team are..."
    
    Example: "What is a language model?"
    Returns: "What is a language model?"
    """
    # Simple split on the first colon.
    if ":" in user_input:
        parts = user_input.split(":", 1)
        # Check if the part before the colon is short (likely a command)
        if len(parts[0]) < 40:
            return parts[1].strip()
    
    # If no colon or the text before it is long, return the whole thing
    return user_input.strip()
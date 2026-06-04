"""
Specialized AI Agents for Interview Analysis
Each agent focuses on a specific aspect of interview evaluation
"""

import logging
from typing import Dict, List, Any
from abc import ABC, abstractmethod
import json

logger = logging.getLogger(__name__)

from rag_engine import GroqLLMInference, RAGRetriever


class InterviewAgent(ABC):
    """Base class for interview analysis agents"""
    
    def __init__(self, llm: GroqLLMInference, retriever: RAGRetriever = None):
        self.llm = llm
        self.retriever = retriever
        self.name = self.__class__.__name__
    
    @abstractmethod
    def analyze(self, *args, **kwargs) -> Dict[str, Any]:
        """Analyze and return findings"""
        pass


class ResumeAnalysisAgent(InterviewAgent):
    """Analyze resume and extract key information"""
    
    def analyze(self, resume_content: str = None) -> Dict[str, Any]:
        """
        Analyze resume content
        
        Returns:
            Dictionary with skills, experience, education, strengths
        """
        prompt = """Analyze the provided resume and extract the following information in JSON format:
        1. Technical Skills (programming languages, frameworks, tools)
        2. Professional Experience (companies, roles, duration)
        3. Education (degrees, institutions)
        4. Key Strengths (based on achievements)
        5. Potential Gaps (areas that could be improved)
        
        Provide structured output as valid JSON."""
        
        try:
            if self.retriever and resume_content is None:
                result = self.llm.generate_with_retrieval(
                    prompt, 
                    self.retriever, 
                    doc_type_filter='resume'
                )
                response = result['response']
                sources = result['retrieved_sources']
            else:
                response = self.llm.generate(prompt, resume_content or "")
                sources = []
            
            # Parse JSON response
            try:
                analysis = json.loads(response)
            except:
                analysis = {'raw_analysis': response}
            
            return {
                'status': 'success',
                'agent': self.name,
                'analysis': analysis,
                'sources': sources
            }
        except Exception as e:
            logger.error(f"Resume analysis error: {e}")
            return {
                'status': 'error',
                'agent': self.name,
                'error': str(e)
            }


class JobDescriptionAgent(InterviewAgent):
    """Analyze job description and extract requirements"""
    
    def analyze(self, jd_content: str = None) -> Dict[str, Any]:
        """
        Analyze job description
        
        Returns:
            Dictionary with required skills, nice-to-have, responsibilities
        """
        prompt = """Analyze the job description and extract in JSON format:
        1. Required Technical Skills
        2. Required Soft Skills
        3. Nice-to-Have Skills
        4. Key Responsibilities
        5. Experience Level Required
        6. Salary/Compensation (if mentioned)
        
        Provide structured output as valid JSON."""
        
        try:
            if self.retriever and jd_content is None:
                result = self.llm.generate_with_retrieval(
                    prompt,
                    self.retriever,
                    doc_type_filter='job_description'
                )
                response = result['response']
                sources = result['retrieved_sources']
            else:
                response = self.llm.generate(prompt, jd_content or "")
                sources = []
            
            try:
                analysis = json.loads(response)
            except:
                analysis = {'raw_analysis': response}
            
            return {
                'status': 'success',
                'agent': self.name,
                'analysis': analysis,
                'sources': sources
            }
        except Exception as e:
            logger.error(f"Job description analysis error: {e}")
            return {
                'status': 'error',
                'agent': self.name,
                'error': str(e)
            }


class SkillGapAgent(InterviewAgent):
    """Detect skill gaps between candidate and job requirements"""
    
    def analyze(self, resume_analysis: Dict, jd_analysis: Dict) -> Dict[str, Any]:
        """
        Compare resume and JD to identify gaps
        
        Args:
            resume_analysis: Output from ResumeAnalysisAgent
            jd_analysis: Output from JobDescriptionAgent
        
        Returns:
            Dictionary with skill gaps, strengths, and recommendations
        """
        prompt = f"""Based on this candidate profile and job requirements, identify:
        
        Candidate Skills:
        {json.dumps(resume_analysis.get('analysis', {}), indent=2)}
        
        Job Requirements:
        {json.dumps(jd_analysis.get('analysis', {}), indent=2)}
        
        Provide JSON analysis with:
        1. Critical Skill Gaps (must-haves missing)
        2. Nice-to-Have Gaps
        3. Matching Strengths
        4. Transferable Skills
        5. Priority Learning Areas"""
        
        try:
            response = self.llm.generate(prompt, max_tokens=1500)
            
            try:
                analysis = json.loads(response)
            except:
                analysis = {'raw_analysis': response}
            
            return {
                'status': 'success',
                'agent': self.name,
                'analysis': analysis
            }
        except Exception as e:
            logger.error(f"Skill gap analysis error: {e}")
            return {
                'status': 'error',
                'agent': self.name,
                'error': str(e)
            }


class CommunicationAnalysisAgent(InterviewAgent):
    """Analyze communication patterns and quality"""
    
    def analyze(self, transcript: str = None, emotion_data: Dict = None, 
               speech_data: Dict = None) -> Dict[str, Any]:
        """
        Analyze communication quality
        
        Returns:
            Dictionary with clarity, confidence, fluency, engagement scores
        """
        context = f"""
        Interview Transcript: {transcript or 'N/A'}
        
        Emotion Data: {json.dumps(emotion_data or {}, indent=2)}
        
        Speech Metrics: {json.dumps(speech_data or {}, indent=2)}
        """
        
        prompt = """Analyze the communication and provide JSON with:
        1. Clarity Score (0-100): How clear and understandable
        2. Confidence Score (0-100): Perceived confidence level
        3. Fluency Score (0-100): Speech flow and pacing
        4. Engagement Score (0-100): Interest and enthusiasm
        5. Filler Words: Frequency and impact
        6. Positive Communication Patterns
        7. Areas for Improvement
        8. Overall Assessment"""
        
        try:
            response = self.llm.generate(prompt, context, max_tokens=1500)
            
            try:
                analysis = json.loads(response)
            except:
                analysis = {'raw_analysis': response}
            
            return {
                'status': 'success',
                'agent': self.name,
                'analysis': analysis
            }
        except Exception as e:
            logger.error(f"Communication analysis error: {e}")
            return {
                'status': 'error',
                'agent': self.name,
                'error': str(e)
            }


class TechnicalCompetencyAgent(InterviewAgent):
    """Evaluate technical competency based on answers"""
    
    def analyze(self, answers: List[str], resume_skills: List[str] = None,
               job_skills: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate technical depth and relevance
        
        Returns:
            Technical proficiency assessment
        """
        prompt = f"""Evaluate the technical competency demonstrated:
        
        Candidate Skills: {json.dumps(resume_skills or [], indent=2)}
        Required Skills: {json.dumps(job_skills or [], indent=2)}
        
        Interview Answers:
        {json.dumps(answers or [], indent=2)}
        
        Provide JSON assessment with:
        1. Technical Depth Score (0-100)
        2. Relevance Score (0-100)
        3. Problem-Solving Ability
        4. Knowledge Depth (Shallow/Moderate/Deep) per technology
        5. Practical Experience Level
        6. Demonstrated Expertise Areas
        7. Knowledge Gaps
        8. Recommendations for improvement"""
        
        try:
            response = self.llm.generate(prompt, max_tokens=1500)
            
            try:
                analysis = json.loads(response)
            except:
                analysis = {'raw_analysis': response}
            
            return {
                'status': 'success',
                'agent': self.name,
                'analysis': analysis
            }
        except Exception as e:
            logger.error(f"Technical competency analysis error: {e}")
            return {
                'status': 'error',
                'agent': self.name,
                'error': str(e)
            }


class BehavioralAssessmentAgent(InterviewAgent):
    """Assess behavioral competencies and soft skills"""
    
    def analyze(self, answers: List[str], emotions: Dict = None) -> Dict[str, Any]:
        """
        Assess behavioral competencies using STAR method
        
        Returns:
            Behavioral assessment with competency scores
        """
        prompt = f"""Assess behavioral competencies from interview responses:
        
        Answers: {json.dumps(answers or [], indent=2)}
        Emotional Data: {json.dumps(emotions or {}, indent=2)}
        
        Evaluate and provide JSON with:
        1. Leadership Score (0-100)
        2. Teamwork Score (0-100)
        3. Problem-Solving Score (0-100)
        4. Adaptability Score (0-100)
        5. Communication Score (0-100)
        6. STAR Method Compliance (% of answers)
        7. Specific Competency Examples
        8. Red Flags (if any)
        9. Strengths Demonstrated
        10. Development Areas"""
        
        try:
            response = self.llm.generate(prompt, max_tokens=1500)
            
            try:
                analysis = json.loads(response)
            except:
                analysis = {'raw_analysis': response}
            
            return {
                'status': 'success',
                'agent': self.name,
                'analysis': analysis
            }
        except Exception as e:
            logger.error(f"Behavioral assessment error: {e}")
            return {
                'status': 'error',
                'agent': self.name,
                'error': str(e)
            }


class FeedbackRecommendationAgent(InterviewAgent):
    """Generate personalized feedback and recommendations"""
    
    def analyze(self, all_analyses: Dict) -> Dict[str, Any]:
        """
        Synthesize all analyses and provide recommendations
        
        Args:
            all_analyses: Dictionary with all agent analysis results
        
        Returns:
            Comprehensive feedback and action plan
        """
        prompt = f"""Based on the comprehensive interview analysis:
        
        {json.dumps(all_analyses, indent=2)}
        
        Provide structured feedback as JSON with:
        1. Overall Assessment (Hire/Strong/Good/Fair/Need Improvement)
        2. Key Strengths (Top 3)
        3. Critical Improvement Areas (Top 3)
        4. Personalized Action Plan:
           - Immediate Actions (next 2 weeks)
           - Short-term Goals (1 month)
           - Long-term Development (3-6 months)
        5. Interview Preparation Tips
        6. Technical Learning Resources
        7. Communication Tips
        8. Next Steps for Candidate
        9. Hiring Recommendation with Confidence (0-100)"""
        
        try:
            response = self.llm.generate(prompt, max_tokens=2000)
            
            try:
                analysis = json.loads(response)
            except:
                analysis = {'raw_analysis': response}
            
            return {
                'status': 'success',
                'agent': self.name,
                'analysis': analysis
            }
        except Exception as e:
            logger.error(f"Feedback generation error: {e}")
            return {
                'status': 'error',
                'agent': self.name,
                'error': str(e)
            }


class InterviewAgentOrchestrator:
    """Orchestrate multiple agents for complete interview analysis"""
    
    def __init__(self, llm: GroqLLMInference, retriever: RAGRetriever = None):
        self.llm = llm
        self.retriever = retriever
        
        # Initialize all agents
        self.resume_agent = ResumeAnalysisAgent(llm, retriever)
        self.jd_agent = JobDescriptionAgent(llm, retriever)
        self.skill_gap_agent = SkillGapAgent(llm, retriever)
        self.communication_agent = CommunicationAnalysisAgent(llm, retriever)
        self.technical_agent = TechnicalCompetencyAgent(llm, retriever)
        self.behavioral_agent = BehavioralAssessmentAgent(llm, retriever)
        self.feedback_agent = FeedbackRecommendationAgent(llm, retriever)
    
    def run_full_analysis(self, resume: str = None, job_description: str = None,
                         interview_transcript: str = None, emotion_data: Dict = None,
                         speech_data: Dict = None, answers: List[str] = None) -> Dict:
        """
        Run complete interview analysis pipeline
        
        Returns:
            Comprehensive analysis report from all agents
        """
        logger.info("Starting full interview analysis pipeline")
        
        results = {
            'timestamp': None,
            'resume_analysis': None,
            'job_analysis': None,
            'skill_gap_analysis': None,
            'communication_analysis': None,
            'technical_analysis': None,
            'behavioral_analysis': None,
            'final_feedback': None
        }
        
        try:
            # Step 1: Analyze resume
            logger.info("Analyzing resume...")
            results['resume_analysis'] = self.resume_agent.analyze(resume)
            
            # Step 2: Analyze job description
            logger.info("Analyzing job description...")
            results['job_analysis'] = self.jd_agent.analyze(job_description)
            
            # Step 3: Detect skill gaps
            logger.info("Detecting skill gaps...")
            results['skill_gap_analysis'] = self.skill_gap_agent.analyze(
                results['resume_analysis'],
                results['job_analysis']
            )
            
            # Step 4: Communication analysis
            logger.info("Analyzing communication...")
            results['communication_analysis'] = self.communication_agent.analyze(
                interview_transcript,
                emotion_data,
                speech_data
            )
            
            # Step 5: Technical competency
            logger.info("Evaluating technical competency...")
            results['technical_analysis'] = self.technical_agent.analyze(
                answers or []
            )
            
            # Step 6: Behavioral assessment
            logger.info("Assessing behavioral competencies...")
            results['behavioral_analysis'] = self.behavioral_agent.analyze(
                answers or [],
                emotion_data
            )
            
            # Step 7: Final feedback and recommendations
            logger.info("Generating final feedback...")
            results['final_feedback'] = self.feedback_agent.analyze(results)
            
            logger.info("Interview analysis pipeline completed successfully")
            return {
                'status': 'success',
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error in analysis pipeline: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'partial_results': results
            }

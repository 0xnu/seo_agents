#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@name: seo_agents.py
@author: Finbarrs Oketunji
@contact: f@finbarrs.eu
@time: Friday June 20 23:15:00 2025
@updated: Sunday June 22 16:34:03 2025
@desc: SEO AI Agents for SEO Analysis.
@version: 0.0.2
@license: MIT
@run: python3 seo_agents.py
"""

import json
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
import time

import openai
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


class AIProvider(Enum):
    """Enumeration of supported AI providers"""
    GPT4 = "gpt-4.1"
    CLAUDE = "claude-3-7-sonnet-20250219"


class SEOTaskType(Enum):
    """Enumeration of SEO task types"""
    KEYWORDS_RESEARCH = "keywords_research"
    LONG_TAIL_SEO = "long_tail_seo"
    SHORT_TAIL_SEO = "short_tail_seo"
    CONTENT_PLANNING = "content_planning"


@dataclass
class SEOResult:
    """Data class for SEO analysis results"""
    task_type: SEOTaskType
    query: str
    results: Dict[str, Any]
    provider: AIProvider
    timestamp: str
    agent_outputs: Dict[str, str]


class AIModelManager:
    """Manages AI model configurations and clients"""
    
    def __init__(self):
        self.openai_client: Optional[openai.OpenAI] = None
        self.openai_llm: Optional[ChatOpenAI] = None
        self.claude_llm: Optional[ChatAnthropic] = None
        self.gpt4_key: Optional[str] = None
        self.claude_key: Optional[str] = None
    
    def configure_openai(self, api_key: str):
        """Configure OpenAI client and LangChain chat model"""
        self.gpt4_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
        self.openai_client = openai.OpenAI(api_key=api_key)
        self.openai_llm = ChatOpenAI(
            model="gpt-4.1",
            temperature=0.7,
            max_tokens=2000,
            api_key=api_key
        )
        print("✓ OpenAI GPT-4 configured")
    
    def configure_claude(self, api_key: str):
        """Configure Claude LangChain chat model"""
        self.claude_key = api_key
        os.environ["ANTHROPIC_API_KEY"] = api_key
        self.claude_llm = ChatAnthropic(
            model="claude-3-7-sonnet-20250219",
            temperature=0.7,
            max_tokens=2000,
            anthropic_api_key=api_key
        )
        print("✓ Claude Sonnet configured")
    
    def get_llm(self, provider: AIProvider):
        """Get LangChain LLM instance for specified provider"""
        if provider == AIProvider.GPT4:
            if not self.openai_llm:
                raise ValueError("OpenAI not configured")
            return self.openai_llm
        elif provider == AIProvider.CLAUDE:
            if not self.claude_llm:
                raise ValueError("Claude not configured")
            return self.claude_llm
        else:
            raise ValueError(f"Unsupported provider: {provider}")


class SEOAgentFactory:
    """Factory class for creating specialised SEO agents"""
    
    @staticmethod
    def create_keywords_researcher(llm) -> Agent:
        """Create keywords research specialist agent"""
        return Agent(
            role="Keywords Research Specialist",
            goal="Conduct comprehensive keyword research and analysis",
            backstory="""You are an expert SEO researcher with 15 years of experience in keyword analysis. 
            You excel at identifying high-value keywords, understanding search intent, and providing actionable 
            insights for SEO strategies. Your expertise includes competitive analysis, search volume estimation, 
            and keyword difficulty assessment.""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
    
    @staticmethod
    def create_long_tail_specialist(llm) -> Agent:
        """Create long-tail SEO specialist agent"""
        return Agent(
            role="Long-tail SEO Specialist",
            goal="Identify and analyse long-tail keyword opportunities",
            backstory="""You are a long-tail SEO expert who specialises in finding low-competition, 
            high-conversion keyword opportunities. You understand the nuances of voice search, 
            question-based queries, and local SEO variations. Your strength lies in discovering 
            untapped keyword niches that competitors overlook.""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
    
    @staticmethod
    def create_short_tail_strategist(llm) -> Agent:
        """Create short-tail SEO strategist agent"""
        return Agent(
            role="Short-tail SEO Strategist",
            goal="Develop strategies for competitive short-tail keywords",
            backstory="""You are a competitive SEO strategist with expertise in high-volume, 
            competitive keywords. You excel at SERP analysis, understanding ranking factors, 
            and developing comprehensive strategies to compete for head terms. Your experience 
            includes working with enterprise-level SEO campaigns.""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
    
    @staticmethod
    def create_content_planner(llm) -> Agent:
        """Create content planning strategist agent"""
        return Agent(
            role="Content Planning Strategist",
            goal="Develop comprehensive content strategies and calendars",
            backstory="""You are a content strategy expert with deep understanding of SEO content planning. 
            You excel at creating content calendars, topic clusters, and aligning content with the customer journey. 
            Your expertise includes content gap analysis, seasonal planning, and multi-format content strategies.""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
    
    @staticmethod
    def create_seo_analyst(llm) -> Agent:
        """Create SEO analyst for quality assurance"""
        return Agent(
            role="SEO Quality Analyst",
            goal="Review and enhance SEO recommendations",
            backstory="""You are a senior SEO analyst who reviews and validates SEO strategies. 
            You ensure recommendations are practical, up-to-date with current SEO best practices, 
            and aligned with business objectives. Your role is to provide quality assurance and 
            strategic oversight.""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )


class SEOTaskFactory:
    """Factory class for creating SEO tasks"""
    
    @staticmethod
    def create_keywords_research_task(topic: str, agent: Agent) -> Task:
        """Create keywords research task"""
        return Task(
            description=f"""
            Conduct comprehensive keyword research for the topic: "{topic}"
            
            Your analysis must include:
            1. Primary keywords (5-10) - High-volume, relevant terms with search volumes
            2. Secondary keywords (10-15) - Medium-volume supporting terms
            3. LSI keywords - Semantically related terms for content depth
            4. Search intent analysis - Classify keywords by intent (informational, navigational, transactional, commercial)
            5. Competition assessment - Evaluate keyword difficulty and competition levels
            6. Opportunity scoring - Rank keywords by potential ROI
            7. SERP feature opportunities - Identify featured snippet, knowledge panel chances
            
            Provide structured output with clear categorisation and actionable recommendations.
            Focus on commercial viability and realistic ranking potential.
            """,
            agent=agent,
            expected_output="Structured keyword research report with categorised keywords, metrics, and strategic recommendations"
        )
    
    @staticmethod
    def create_long_tail_task(topic: str, agent: Agent) -> Task:
        """Create long-tail SEO analysis task"""
        return Task(
            description=f"""
            Analyse long-tail SEO opportunities for: "{topic}"
            
            Deliver comprehensive analysis including:
            1. Long-tail keyword variations (25-30 phrases, 3+ words each)
            2. Question-based keywords - Who, what, when, where, why, how variations
            3. Conversational queries - Natural language and voice search optimised
            4. Local SEO variations - Geographic modifiers where applicable
            5. Buyer intent phrases - Purchase-ready long-tail keywords
            6. Content cluster opportunities - Related topic groupings
            7. Featured snippet targets - Question-answer format opportunities
            8. Competitor gap analysis - Missed opportunities in competitor content
            
            Prioritise keywords with lower competition but meaningful search volume.
            Include conversion potential assessment for each keyword group.
            """,
            agent=agent,
            expected_output="Comprehensive long-tail keyword strategy with clustered opportunities and conversion potential analysis"
        )
    
    @staticmethod
    def create_short_tail_task(topic: str, agent: Agent) -> Task:
        """Create short-tail SEO strategy task"""
        return Task(
            description=f"""
            Develop short-tail SEO strategy for: "{topic}"
            
            Provide strategic analysis covering:
            1. High-volume head terms (1-2 words) - Primary targets with volume data
            2. Competitive landscape analysis - Top 10 competitors for each term
            3. SERP analysis - Current ranking factors and feature opportunities
            4. Content pillar strategy - Hub and spoke content architecture
            5. Authority building requirements - Backlink and domain authority needs
            6. Technical SEO considerations - Page speed, mobile, structured data
            7. Ranking timeline estimates - Realistic timeframes for results
            8. Resource requirements - Content, links, technical resources needed
            
            Focus on achievable strategies for competing against established players.
            Include both short-term quick wins and long-term authority building.
            """,
            agent=agent,
            expected_output="Strategic short-tail SEO plan with competitive analysis and resource requirements"
        )
    
    @staticmethod
    def create_content_planning_task(topic: str, agent: Agent) -> Task:
        """Create content planning strategy task"""
        return Task(
            description=f"""
            Create comprehensive content planning strategy for: "{topic}"
            
            Develop detailed strategy including:
            1. Content calendar - 12-month strategic timeline with themes
            2. Content types and formats - Blogs, guides, videos, infographics, tools
            3. Topic clusters - Pillar pages and supporting content architecture
            4. Customer journey mapping - Content for awareness, consideration, decision stages
            5. Seasonal content opportunities - Trending topics and timely content
            6. Content distribution strategy - Channels and promotion tactics
            7. Internal linking strategy - Site architecture and link flow
            8. Content gaps analysis - Opportunities competitors are missing
            9. Performance metrics - KPIs and measurement framework
            10. Content production workflow - Team roles and processes
            
            Align content strategy with business objectives and SEO goals.
            Include both evergreen content and trending topic opportunities.
            """,
            agent=agent,
            expected_output="Complete content strategy with calendar, architecture, and implementation roadmap"
        )
    
    @staticmethod
    def create_quality_review_task(primary_output: str, agent: Agent) -> Task:
        """Create quality review and enhancement task"""
        return Task(
            description=f"""
            Review and enhance the SEO analysis provided:
            
            {primary_output}
            
            Your quality review should:
            1. Validate recommendations against current SEO best practices
            2. Identify any gaps or missing opportunities
            3. Enhance strategic recommendations with specific implementation steps
            4. Provide priority ranking for all recommendations
            5. Add competitive intelligence insights
            6. Include potential risks and mitigation strategies
            7. Suggest success metrics and measurement approaches
            
            Ensure all recommendations are practical, actionable, and ROI-focused.
            """,
            agent=agent,
            expected_output="Enhanced SEO strategy with validated recommendations and implementation priorities"
        )


class SEOCrewManager:
    """Manages CrewAI crews for different SEO tasks"""
    
    def __init__(self, model_manager: AIModelManager):
        self.model_manager = model_manager
        self.agent_factory = SEOAgentFactory()
        self.task_factory = SEOTaskFactory()
    
    def execute_keywords_research(self, topic: str, provider: AIProvider) -> Dict[str, str]:
        """Execute keywords research using CrewAI"""
        llm = self.model_manager.get_llm(provider)
        
        # Create agents
        researcher = self.agent_factory.create_keywords_researcher(llm)
        analyst = self.agent_factory.create_seo_analyst(llm)
        
        # Create tasks
        research_task = self.task_factory.create_keywords_research_task(topic, researcher)
        
        # Create and execute crew
        crew = Crew(
            agents=[researcher, analyst],
            tasks=[research_task],
            verbose=True,
            process=Process.sequential
        )
        
        result = crew.kickoff()
        
        return {
            "primary_analysis": str(result),
            "agent_contributions": {
                "keywords_researcher": "Comprehensive keyword research and analysis",
                "seo_analyst": "Strategic validation and enhancement"
            }
        }
    
    def execute_long_tail_analysis(self, topic: str, provider: AIProvider) -> Dict[str, str]:
        """Execute long-tail SEO analysis using CrewAI"""
        llm = self.model_manager.get_llm(provider)
        
        # Create agents
        specialist = self.agent_factory.create_long_tail_specialist(llm)
        analyst = self.agent_factory.create_seo_analyst(llm)
        
        # Create tasks
        analysis_task = self.task_factory.create_long_tail_task(topic, specialist)
        
        # Create and execute crew
        crew = Crew(
            agents=[specialist, analyst],
            tasks=[analysis_task],
            verbose=True,
            process=Process.sequential
        )
        
        result = crew.kickoff()
        
        return {
            "primary_analysis": str(result),
            "agent_contributions": {
                "long_tail_specialist": "Long-tail keyword opportunity analysis",
                "seo_analyst": "Strategic validation and prioritisation"
            }
        }
    
    def execute_short_tail_strategy(self, topic: str, provider: AIProvider) -> Dict[str, str]:
        """Execute short-tail SEO strategy using CrewAI"""
        llm = self.model_manager.get_llm(provider)
        
        # Create agents
        strategist = self.agent_factory.create_short_tail_strategist(llm)
        analyst = self.agent_factory.create_seo_analyst(llm)
        
        # Create tasks
        strategy_task = self.task_factory.create_short_tail_task(topic, strategist)
        
        # Create and execute crew
        crew = Crew(
            agents=[strategist, analyst],
            tasks=[strategy_task],
            verbose=True,
            process=Process.sequential
        )
        
        result = crew.kickoff()
        
        return {
            "primary_analysis": str(result),
            "agent_contributions": {
                "short_tail_strategist": "Competitive short-tail keyword strategy",
                "seo_analyst": "Strategic validation and resource planning"
            }
        }
    
    def execute_content_planning(self, topic: str, provider: AIProvider) -> Dict[str, str]:
        """Execute content planning using CrewAI"""
        llm = self.model_manager.get_llm(provider)
        
        # Create agents
        planner = self.agent_factory.create_content_planner(llm)
        analyst = self.agent_factory.create_seo_analyst(llm)
        
        # Create tasks
        planning_task = self.task_factory.create_content_planning_task(topic, planner)
        
        # Create and execute crew
        crew = Crew(
            agents=[planner, analyst],
            tasks=[planning_task],
            verbose=True,
            process=Process.sequential
        )
        
        result = crew.kickoff()
        
        return {
            "primary_analysis": str(result),
            "agent_contributions": {
                "content_planner": "Comprehensive content strategy and calendar",
                "seo_analyst": "Strategic validation and implementation roadmap"
            }
        }


class SEOAgent:
    """Main SEO Agent class using CrewAI for multi-agent coordination"""
    
    def __init__(self):
        self.model_manager = AIModelManager()
        self.crew_manager = SEOCrewManager(self.model_manager)
        self.results_history: List[SEOResult] = []
    
    def configure_api_keys(self, gpt4_key: Optional[str] = None, claude_key: Optional[str] = None):
        """Configure API keys for AI providers"""
        if gpt4_key:
            self.model_manager.configure_openai(gpt4_key)
        
        if claude_key:
            self.model_manager.configure_claude(claude_key)
    
    def execute_seo_task(self, task_type: SEOTaskType, topic: str, provider: AIProvider) -> SEOResult:
        """Execute SEO task using CrewAI multi-agent system"""
        print(f"Executing {task_type.value} for '{topic}' using {provider.value}...")
        print("Initialising CrewAI agents...")
        
        try:
            # Execute task based on type
            if task_type == SEOTaskType.KEYWORDS_RESEARCH:
                agent_outputs = self.crew_manager.execute_keywords_research(topic, provider)
            elif task_type == SEOTaskType.LONG_TAIL_SEO:
                agent_outputs = self.crew_manager.execute_long_tail_analysis(topic, provider)
            elif task_type == SEOTaskType.SHORT_TAIL_SEO:
                agent_outputs = self.crew_manager.execute_short_tail_strategy(topic, provider)
            elif task_type == SEOTaskType.CONTENT_PLANNING:
                agent_outputs = self.crew_manager.execute_content_planning(topic, provider)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
            
            # Create timestamp for file naming
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Create result object
            result = SEOResult(
                task_type=task_type,
                query=topic,
                results={"analysis": agent_outputs["primary_analysis"]},
                provider=provider,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                agent_outputs=agent_outputs["agent_contributions"]
            )
            
            # Store in history
            self.results_history.append(result)
            
            # Write result to timestamped file
            self._write_result_to_file(result, timestamp)
            
            print(f"✓ Task completed successfully with {len(agent_outputs['agent_contributions'])} agents")
            print(f"✓ Results saved to result_{timestamp}.txt")
            return result
            
        except Exception as e:
            print(f"✗ Task failed: {str(e)}")
            raise
    
    def keywords_research(self, topic: str, provider: AIProvider) -> SEOResult:
        """Perform keywords research analysis"""
        return self.execute_seo_task(SEOTaskType.KEYWORDS_RESEARCH, topic, provider)
    
    def long_tail_seo(self, topic: str, provider: AIProvider) -> SEOResult:
        """Perform long-tail SEO analysis"""
        return self.execute_seo_task(SEOTaskType.LONG_TAIL_SEO, topic, provider)
    
    def short_tail_seo(self, topic: str, provider: AIProvider) -> SEOResult:
        """Perform short-tail SEO analysis"""
        return self.execute_seo_task(SEOTaskType.SHORT_TAIL_SEO, topic, provider)
    
    def content_planning(self, topic: str, provider: AIProvider) -> SEOResult:
        """Perform content planning analysis"""
        return self.execute_seo_task(SEOTaskType.CONTENT_PLANNING, topic, provider)
    
    def _write_result_to_file(self, result: SEOResult, timestamp: str):
        """Write result to timestamped text file"""
        filename = f"result_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # Write header
                f.write("="*80 + "\n")
                f.write("SEO AI AGENT ANALYSIS REPORT\n")
                f.write("="*80 + "\n\n")
                
                # Write metadata
                f.write(f"Task Type: {result.task_type.value.replace('_', ' ').title()}\n")
                f.write(f"Query/Topic: {result.query}\n")
                f.write(f"AI Provider: {result.provider.value}\n")
                f.write(f"Analysis Date: {result.timestamp}\n")
                f.write(f"Report File: {filename}\n\n")
                
                # Write agent contributions
                f.write("AGENT CONTRIBUTIONS\n")
                f.write("-"*40 + "\n")
                for agent, contribution in result.agent_outputs.items():
                    f.write(f"• {agent.replace('_', ' ').title()}:\n")
                    f.write(f"  {contribution}\n\n")
                
                # Write main analysis
                f.write("DETAILED ANALYSIS\n")
                f.write("-"*40 + "\n")
                f.write(result.results['analysis'])
                f.write("\n\n")
                
                # Write footer
                f.write("="*80 + "\n")
                f.write("End of SEO Analysis Report\n")
                f.write(f"Generated by Finn SEO Agent on {result.timestamp}\n")
                f.write("="*80 + "\n")
                
        except Exception as e:
            print(f"Warning: Failed to write result to file {filename}: {str(e)}")
    
    def export_results(self, filename: str = "seo_results.json"):
        """Export all results to JSON file"""
        if not self.results_history:
            print("No results to export")
            return
        
        export_data = []
        for result in self.results_history:
            export_data.append({
                "task_type": result.task_type.value,
                "query": result.query,
                "results": result.results,
                "provider": result.provider.value,
                "timestamp": result.timestamp,
                "agent_contributions": result.agent_outputs
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results exported to {filename}")
    
    def display_result(self, result: SEOResult):
        """Display formatted result with agent contributions"""
        print(f"\n{'='*80}")
        print(f"Task: {result.task_type.value.replace('_', ' ').title()}")
        print(f"Query: {result.query}")
        print(f"Provider: {result.provider.value}")
        print(f"Timestamp: {result.timestamp}")
        print(f"{'='*80}")
        
        # Display agent contributions
        print("Agent Contributions:")
        for agent, contribution in result.agent_outputs.items():
            print(f"  • {agent.replace('_', ' ').title()}: {contribution}")
        print(f"{'='*80}")
        
        # Display main analysis
        print("Analysis Results:")
        print(result.results['analysis'])
        print(f"{'='*80}\n")


class SEOAgentCLI:
    """Command-line interface for the CrewAI-powered SEO Agent"""
    
    def __init__(self):
        self.agent = SEOAgent()
        self.setup_complete = False
    
    def setup_api_keys(self):
        """Interactive API key setup"""
        print("Finn SEO Agent Setup")
        print("=" * 50)
        print("Configure your AI providers:")
        
        gpt4_key = input("Enter OpenAI API key (or press Enter to skip): ").strip()
        claude_key = input("Enter Claude API key (or press Enter to skip): ").strip()
        
        if not gpt4_key and not claude_key:
            print("✗ At least one API key is required")
            return False
        
        self.agent.configure_api_keys(
            gpt4_key if gpt4_key else None,
            claude_key if claude_key else None
        )
        
        self.setup_complete = True
        print("✓ Setup completed successfully")
        return True
    
    def display_menu(self):
        """Display main menu options"""
        print("\nFinn SEO Agent - Main Menu")
        print("=" * 40)
        print("1. Keywords Research (Multi-agent)")
        print("2. Long-tail SEO (Multi-agent)")
        print("3. Short-tail SEO (Multi-agent)")
        print("4. Content Planning (Multi-agent)")
        print("5. Export Results (JSON)")
        print("6. View Results History")
        print("7. List Result Files")
        print("8. Exit")
        print("=" * 40)
    
    def get_provider_choice(self) -> AIProvider:
        """Get AI provider choice from user"""
        available_providers = []
        
        if self.agent.model_manager.openai_llm:
            available_providers.append((1, AIProvider.GPT4))
        if self.agent.model_manager.claude_llm:
            available_providers.append((2, AIProvider.CLAUDE))
        
        print("\nSelect AI Provider for CrewAI Agents:")
        for idx, (num, provider) in enumerate(available_providers):
            print(f"{num}. {provider.value}")
        
        while True:
            try:
                choice = int(input("Choice: "))
                for num, provider in available_providers:
                    if choice == num:
                        return provider
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    def run_task(self, task_type: SEOTaskType):
        """Run a specific SEO task with multi-agent system"""
        topic = input(f"Enter topic for {task_type.value.replace('_', ' ')}: ").strip()
        if not topic:
            print("Topic cannot be empty")
            return
        
        provider = self.get_provider_choice()
        
        print("\nInitialising CrewAI multi-agent system...")
        print(f"Task: {task_type.value.replace('_', ' ').title()}")
        print(f"Topic: {topic}")
        print(f"Provider: {provider.value}")
        print("This may take a few minutes as multiple agents collaborate...")
        
        try:
            result = self.agent.execute_seo_task(task_type, topic, provider)
            self.agent.display_result(result)
        except Exception as e:
            print(f"Error executing task: {str(e)}")
    
    def list_result_files(self):
        """List all result files in current directory"""
        result_files = [f for f in os.listdir('.') if f.startswith('result_') and f.endswith('.txt')]
        
        if not result_files:
            print("No result files found in current directory")
            return
        
        result_files.sort(reverse=True)  # Sort by newest first
        
        print(f"\nFound {len(result_files)} result files:")
        print("=" * 50)
        
        for i, filename in enumerate(result_files, 1):
            # Extract timestamp from filename
            timestamp_part = filename.replace('result_', '').replace('.txt', '')
            try:
                # Convert timestamp to readable format
                timestamp_obj = time.strptime(timestamp_part, "%Y%m%d_%H%M%S")
                readable_time = time.strftime("%Y-%m-%d %H:%M:%S", timestamp_obj)
                file_size = os.path.getsize(filename)
                print(f"{i:2d}. {filename} ({readable_time}) - {file_size} bytes")
            except ValueError:
                print(f"{i:2d}. {filename} (Invalid timestamp format)")
        
        print("=" * 50)
    
    def view_history(self):
        """Display results history"""
        if not self.agent.results_history:
            print("No results in history")
            return
        
        print(f"\nResults History ({len(self.agent.results_history)} items)")
        print("=" * 70)
        
        for i, result in enumerate(self.agent.results_history, 1):
            agents_count = len(result.agent_outputs)
            print(f"{i}. {result.task_type.value} - {result.query} ({result.provider.value}) - {agents_count} agents - {result.timestamp}")
        
        try:
            choice = int(input("\nEnter result number to view (0 to go back): "))
            if 1 <= choice <= len(self.agent.results_history):
                self.agent.display_result(self.agent.results_history[choice - 1])
        except ValueError:
            pass
    
    def run(self):
        """Run the CLI application"""
        print("Welcome to Finn SEO Agent")
        print("Multi-Agent SEO Analysis System")
        print("=" * 50)
        
        if not self.setup_api_keys():
            return
        
        while True:
            self.display_menu()
            
            try:
                choice = int(input("Select option: "))
                
                if choice == 1:
                    self.run_task(SEOTaskType.KEYWORDS_RESEARCH)
                elif choice == 2:
                    self.run_task(SEOTaskType.LONG_TAIL_SEO)
                elif choice == 3:
                    self.run_task(SEOTaskType.SHORT_TAIL_SEO)
                elif choice == 4:
                    self.run_task(SEOTaskType.CONTENT_PLANNING)
                elif choice == 5:
                    filename = input("Export filename (default: seo_results.json): ").strip()
                    self.agent.export_results(filename if filename else "seo_results.json")
                elif choice == 6:
                    self.view_history()
                elif choice == 7:
                    self.list_result_files()
                elif choice == 8:
                    print("Goodbye!")
                    break
                else:
                    print("Invalid option. Please try again.")
                    
            except ValueError:
                print("Please enter a valid number.")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break


def main():
    """Main application entry point"""
    try:
        cli = SEOAgentCLI()
        cli.run()
    except Exception as e:
        print(f"Application error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

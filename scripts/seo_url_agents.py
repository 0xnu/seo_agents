#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@name: seo_url_agents.py
@author: Finbarrs Oketunji
@contact: f@finbarrs.eu
@time: Monday June 23 01:30:00 2025
@updated: Tuesday June 24 01:22:00 2025
@desc: SEO AI Agents for URL-based SEO Analysis.
@version: 0.0.1
@license: MIT
@run: python3 seo_url_agents.py
"""

import json
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
import time
import requests
import urllib.parse
from bs4 import BeautifulSoup
import markdownify

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
    COMPREHENSIVE_ANALYSIS = "comprehensive_analysis"


@dataclass
class PageContent:
    """Data class for extracted page content"""
    url: str
    title: str
    meta_description: str
    headings: Dict[str, List[str]]
    content_markdown: str
    content_json: Dict[str, Any]
    word_count: int
    images: List[str]
    links: Dict[str, List[str]]
    extraction_timestamp: str


@dataclass
class SEOResult:
    """Data class for SEO analysis results"""
    task_type: SEOTaskType
    url: str
    page_content: PageContent
    results: Dict[str, Any]
    provider: AIProvider
    timestamp: str
    agent_outputs: Dict[str, str]


class WebContentExtractor:
    """Extracts and processes web page content"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def extract_page_content(self, url: str) -> PageContent:
        """Extract comprehensive content from a webpage"""
        print(f"Extracting content from: {url}")
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "aside"]):
                script.decompose()
            
            # Extract basic metadata
            title = self._extract_title(soup)
            meta_description = self._extract_meta_description(soup)
            
            # Extract headings
            headings = self._extract_headings(soup)
            
            # Extract main content
            main_content = self._extract_main_content(soup)
            
            # Convert to markdown
            content_markdown = markdownify.markdownify(str(main_content), heading_style="ATX")
            
            # Create JSON structure
            content_json = self._create_content_json(soup, title, meta_description, headings)
            
            # Extract images and links
            images = self._extract_images(soup, url)
            links = self._extract_links(soup, url)
            
            # Calculate word count
            word_count = len(content_markdown.split())
            
            page_content = PageContent(
                url=url,
                title=title,
                meta_description=meta_description,
                headings=headings,
                content_markdown=content_markdown,
                content_json=content_json,
                word_count=word_count,
                images=images,
                links=links,
                extraction_timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
            
            print(f"✓ Content extracted: {word_count} words, {len(images)} images, {sum(len(v) for v in links.values())} links")
            return page_content
            
        except Exception as e:
            print(f"✗ Failed to extract content from {url}: {str(e)}")
            raise
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        h1_tag = soup.find('h1')
        if h1_tag:
            return h1_tag.get_text().strip()
        
        return "No title found"
    
    def _extract_meta_description(self, soup: BeautifulSoup) -> str:
        """Extract meta description"""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            return meta_desc.get('content').strip()
        
        meta_desc = soup.find('meta', attrs={'property': 'og:description'})
        if meta_desc and meta_desc.get('content'):
            return meta_desc.get('content').strip()
        
        return "No meta description found"
    
    def _extract_headings(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Extract all headings organised by level"""
        headings = {f'h{i}': [] for i in range(1, 7)}
        
        for i in range(1, 7):
            heading_tags = soup.find_all(f'h{i}')
            headings[f'h{i}'] = [tag.get_text().strip() for tag in heading_tags if tag.get_text().strip()]
        
        return headings
    
    def _extract_main_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Extract main content, prioritising article, main, or content areas"""
        # Try to find main content areas
        main_selectors = [
            'main',
            'article',
            '[role="main"]',
            '.content',
            '.main-content',
            '.post-content',
            '.entry-content',
            '#content',
            '#main'
        ]
        
        for selector in main_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                return main_content
        
        # Fallback to body content
        body = soup.find('body')
        return body if body else soup
    
    def _create_content_json(self, soup: BeautifulSoup, title: str, meta_description: str, headings: Dict[str, List[str]]) -> Dict[str, Any]:
        """Create structured JSON representation of content"""
        # Extract paragraphs
        paragraphs = [p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip()]
        
        # Extract lists
        lists = []
        for ul in soup.find_all(['ul', 'ol']):
            list_items = [li.get_text().strip() for li in ul.find_all('li') if li.get_text().strip()]
            if list_items:
                lists.append({
                    'type': ul.name,
                    'items': list_items
                })
        
        # Extract structured data
        structured_data = []
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                data = json.loads(script.string)
                structured_data.append(data)
            except (json.JSONDecodeError, TypeError):
                continue
        
        return {
            'title': title,
            'meta_description': meta_description,
            'headings': headings,
            'paragraphs': paragraphs[:50],  # Limit to first 50 paragraphs
            'lists': lists[:20],  # Limit to first 20 lists
            'structured_data': structured_data,
            'content_summary': {
                'paragraph_count': len(paragraphs),
                'list_count': len(lists),
                'heading_distribution': {k: len(v) for k, v in headings.items() if v}
            }
        }
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract image URLs"""
        images = []
        img_tags = soup.find_all('img')
        
        for img in img_tags:
            src = img.get('src') or img.get('data-src')
            if src:
                # Convert relative URLs to absolute
                absolute_url = urllib.parse.urljoin(base_url, src)
                images.append({
                    'url': absolute_url,
                    'alt': img.get('alt', ''),
                    'title': img.get('title', '')
                })
        
        return images[:50]  # Limit to 50 images
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> Dict[str, List[str]]:
        """Extract internal and external links"""
        links = {'internal': [], 'external': []}
        base_domain = urllib.parse.urlparse(base_url).netloc
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            absolute_url = urllib.parse.urljoin(base_url, href)
            parsed_url = urllib.parse.urlparse(absolute_url)
            
            link_data = {
                'url': absolute_url,
                'text': a_tag.get_text().strip(),
                'title': a_tag.get('title', '')
            }
            
            if parsed_url.netloc == base_domain:
                links['internal'].append(link_data)
            elif parsed_url.netloc:  # External link with domain
                links['external'].append(link_data)
        
        # Limit links
        links['internal'] = links['internal'][:100]
        links['external'] = links['external'][:50]
        
        return links


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
            goal="Analyse webpage content and conduct comprehensive keyword research",
            backstory="""You are an expert SEO researcher with 15 years of experience in keyword analysis. 
            You excel at analysing existing webpage content to identify current keyword focus, gaps in keyword targeting, 
            and opportunities for improvement. Your expertise includes competitive analysis based on page content, 
            search intent evaluation, and providing actionable keyword recommendations.""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
    
    @staticmethod
    def create_long_tail_specialist(llm) -> Agent:
        """Create long-tail SEO specialist agent"""
        return Agent(
            role="Long-tail SEO Specialist",
            goal="Identify long-tail keyword opportunities based on webpage content analysis",
            backstory="""You are a long-tail SEO expert who specialises in analysing webpage content to find 
            untapped long-tail keyword opportunities. You understand how to extract semantic meaning from existing content 
            and identify natural language variations, question-based queries, and conversational search terms that align 
            with the page's topic and user intent.""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
    
    @staticmethod
    def create_short_tail_strategist(llm) -> Agent:
        """Create short-tail SEO strategist agent"""
        return Agent(
            role="Short-tail SEO Strategist",
            goal="Develop competitive short-tail keyword strategies based on content analysis",
            backstory="""You are a competitive SEO strategist who analyses webpage content to identify opportunities 
            for high-volume, competitive keywords. You excel at understanding content authority, topic coverage depth, 
            and developing strategies to compete for head terms while maintaining content relevance and quality.""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
    
    @staticmethod
    def create_content_planner(llm) -> Agent:
        """Create content planning strategist agent"""
        return Agent(
            role="Content Planning Strategist",
            goal="Develop content strategies based on existing webpage analysis",
            backstory="""You are a content strategy expert who analyses existing webpage content to identify gaps, 
            improvement opportunities, and expansion possibilities. You excel at creating content roadmaps that build 
            upon existing content authority while addressing user needs and search intent patterns revealed through 
            content analysis.""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )
    
    @staticmethod
    def create_seo_analyst(llm) -> Agent:
        """Create SEO analyst for quality assurance"""
        return Agent(
            role="SEO Quality Analyst",
            goal="Review webpage content and validate SEO recommendations",
            backstory="""You are a senior SEO analyst who reviews webpage content and validates SEO strategies 
            based on actual page performance indicators. You ensure recommendations are practical, aligned with 
            existing content quality, and focused on realistic improvement opportunities based on current page authority 
            and content depth.""",
            verbose=True,
            allow_delegation=False,
            llm=llm
        )


class SEOTaskFactory:
    """Factory class for creating SEO tasks"""
    
    @staticmethod
    def create_keywords_research_task(page_content: PageContent, agent: Agent) -> Task:
        """Create keywords research task based on webpage content"""
        content_summary = f"""
        URL: {page_content.url}
        Title: {page_content.title}
        Meta Description: {page_content.meta_description}
        Word Count: {page_content.word_count}
        
        Headings Structure:
        {json.dumps(page_content.headings, indent=2)}
        
        Content Preview (First 2000 characters):
        {page_content.content_markdown[:2000]}
        
        Content Analysis Data:
        {json.dumps(page_content.content_json['content_summary'], indent=2)}
        """
        
        return Task(
            description=f"""
            Analyse the provided webpage content and conduct comprehensive keyword research:
            
            {content_summary}
            
            Your analysis must include:
            1. Current keyword focus analysis - What keywords is this page currently targeting?
            2. Keyword gap analysis - What relevant keywords are missing from the content?
            3. Primary keyword opportunities (5-10) - High-volume, relevant terms with search volumes
            4. Secondary keyword opportunities (10-15) - Medium-volume supporting terms
            5. LSI keywords - Semantically related terms that could strengthen content depth
            6. Search intent analysis - Classify current and potential keywords by intent
            7. Content-keyword alignment assessment - How well does current content match keyword intent?
            8. Competitive keyword opportunities - Keywords competitors might be targeting for similar content
            9. Improvement recommendations - Specific content modifications to better target keywords
            
            Base your analysis on the actual webpage content provided, not generic keyword research.
            Focus on realistic opportunities that align with the existing content structure and authority.
            """,
            agent=agent,
            expected_output="Detailed keyword analysis report based on webpage content with current focus assessment and improvement opportunities"
        )
    
    @staticmethod
    def create_long_tail_task(page_content: PageContent, agent: Agent) -> Task:
        """Create long-tail SEO analysis task based on webpage content"""
        content_summary = f"""
        URL: {page_content.url}
        Title: {page_content.title}
        Content Topics: {', '.join([h for headings in page_content.headings.values() for h in headings[:5]])}
        Word Count: {page_content.word_count}
        
        Key Content Sections:
        {page_content.content_markdown[:1500]}
        """
        
        return Task(
            description=f"""
            Analyse the webpage content to identify long-tail keyword opportunities:
            
            {content_summary}
            
            Deliver comprehensive long-tail analysis including:
            1. Content-derived long-tail variations (25-30 phrases, 3+ words each) based on actual page topics
            2. Question-based keywords that align with the content's informational value
            3. Conversational queries that natural language processing would associate with this content
            4. Topic-specific long-tail opportunities that build on existing content themes
            5. User intent long-tail phrases that complement the page's current purpose
            6. Content expansion long-tail keywords - opportunities to deepen existing topics
            7. Related topic long-tail keywords that could justify content additions
            8. Local/specific modifier opportunities where applicable to the content
            9. Long-tail keyword priority ranking based on content relevance and search potential
            
            Focus on long-tail opportunities that naturally extend from the existing content rather than generic suggestions.
            Ensure all recommendations align with the page's established topic authority and user intent.
            """,
            agent=agent,
            expected_output="Content-based long-tail keyword strategy with natural extension opportunities and relevance scoring"
        )
    
    @staticmethod
    def create_short_tail_task(page_content: PageContent, agent: Agent) -> Task:
        """Create short-tail SEO strategy task based on webpage content"""
        content_summary = f"""
        URL: {page_content.url}
        Title: {page_content.title}
        Meta Description: {page_content.meta_description}
        Content Authority Indicators: {page_content.word_count} words, {len([h for headings in page_content.headings.values() for h in headings])} headings
        
        Content Structure Analysis:
        {json.dumps(page_content.content_json['content_summary'], indent=2)}
        
        Main Topics Covered:
        {page_content.content_markdown[:1000]}
        """
        
        return Task(
            description=f"""
            Develop short-tail SEO strategy based on webpage content analysis:
            
            {content_summary}
            
            Provide strategic analysis covering:
            1. Current short-tail keyword targeting assessment - What head terms is this page competing for?
            2. Content authority evaluation - Does this page have sufficient depth for competitive short-tail terms?
            3. Short-tail opportunities (5-8 terms, 1-2 words) that align with content strength
            4. Content gap analysis for short-tail competition - What's missing to compete effectively?
            5. Page authority requirements - Assessment of current page strength vs. competitive requirements
            6. Content enhancement strategy - Specific improvements needed to compete for short-tail terms
            7. Internal linking opportunities to boost short-tail keyword authority
            8. Technical SEO considerations based on current page structure
            9. Realistic timeline for short-tail keyword ranking improvements
            10. Resource requirements - Content additions, backlinks, technical improvements needed
            
            Base recommendations on actual page content quality and current competitive positioning.
            Provide honest assessment of short-tail keyword feasibility given existing content authority.
            """,
            agent=agent,
            expected_output="Realistic short-tail SEO strategy based on current page authority with specific improvement roadmap"
        )
    
    @staticmethod
    def create_content_planning_task(page_content: PageContent, agent: Agent) -> Task:
        """Create content planning strategy task based on webpage content"""
        content_summary = f"""
        URL: {page_content.url}
        Title: {page_content.title}
        Current Content Structure: {page_content.word_count} words, {json.dumps(page_content.content_json['content_summary'], indent=2)}
        
        Existing Content Topics:
        Headings: {json.dumps(page_content.headings, indent=2)}
        
        Content Quality Indicators:
        - Images: {len(page_content.images)}
        - Internal Links: {len(page_content.links['internal'])}
        - External Links: {len(page_content.links['external'])}
        
        Current Content Preview:
        {page_content.content_markdown[:2000]}
        """
        
        return Task(
            description=f"""
            Create comprehensive content planning strategy based on existing webpage analysis:
            
            {content_summary}
            
            Develop detailed strategy including:
            1. Content audit results - Strengths and weaknesses of current page content
            2. Content gap analysis - Missing topics, insufficient depth areas, outdated information
            3. Content enhancement priorities - Immediate improvements to existing content
            4. Related content opportunities - New pages/posts that could support this page's authority
            5. Content cluster strategy - How this page fits into broader topic authority building
            6. User journey optimization - Content improvements to better serve user intent at different stages
            7. Content format recommendations - Additional formats (videos, infographics, tools) that could enhance the page
            8. Internal linking strategy - Content connections to improve site architecture
            9. Content update schedule - Regular maintenance and refresh requirements
            10. Performance metrics - KPIs to measure content strategy success based on current baseline
            11. Content distribution strategy - How to promote content improvements
            12. Competitive content analysis - How current content compares to top-ranking competitors
            
            Base all recommendations on the actual page content provided and realistic improvement opportunities.
            Focus on enhancing existing content authority rather than complete content overhauls.
            """,
            agent=agent,
            expected_output="Comprehensive content improvement strategy with specific actionable recommendations based on current page analysis"
        )
    
    @staticmethod
    def create_comprehensive_analysis_task(page_content: PageContent, agent: Agent) -> Task:
        """Create comprehensive SEO analysis task combining all aspects"""
        content_summary = f"""
        URL: {page_content.url}
        Title: {page_content.title}
        Meta Description: {page_content.meta_description}
        Word Count: {page_content.word_count}
        Extraction Time: {page_content.extraction_timestamp}
        
        Full Content Structure:
        {json.dumps(page_content.content_json, indent=2)[:3000]}
        
        Complete Content (First 3000 characters):
        {page_content.content_markdown[:3000]}
        """
        
        return Task(
            description=f"""
            Conduct comprehensive SEO analysis of the provided webpage:
            
            {content_summary}
            
            Provide complete analysis covering:
            
            KEYWORD ANALYSIS:
            1. Current keyword targeting assessment
            2. Primary keyword opportunities (10-15 terms)
            3. Secondary keyword opportunities (15-20 terms)
            4. Long-tail keyword opportunities (30-40 phrases)
            5. Short-tail competitive assessment
            6. Keyword gap analysis
            
            CONTENT ANALYSIS:
            7. Content quality and depth assessment
            8. Topic coverage evaluation
            9. Content structure optimization recommendations
            10. User intent alignment analysis
            
            STRATEGIC RECOMMENDATIONS:
            11. Priority improvement areas
            12. Content enhancement strategy  
            13. Technical SEO considerations
            14. Competitive positioning strategy
            15. Implementation timeline with phases
            16. Success metrics and KPIs
            17. Resource requirements assessment
            
            Base all recommendations on the actual webpage content provided.
            Provide actionable insights that build upon existing content strengths.
            Include specific, measurable improvement suggestions.
            """,
            agent=agent,
            expected_output="Complete SEO analysis report with keyword research, content assessment, and strategic implementation roadmap"
        )


class SEOCrewManager:
    """Manages CrewAI crews for different SEO tasks"""
    
    def __init__(self, model_manager: AIModelManager):
        self.model_manager = model_manager
        self.agent_factory = SEOAgentFactory()
        self.task_factory = SEOTaskFactory()
    
    def execute_keywords_research(self, page_content: PageContent, provider: AIProvider) -> Dict[str, str]:
        """Execute keywords research using CrewAI based on webpage content"""
        llm = self.model_manager.get_llm(provider)
        
        researcher = self.agent_factory.create_keywords_researcher(llm)
        analyst = self.agent_factory.create_seo_analyst(llm)
        
        research_task = self.task_factory.create_keywords_research_task(page_content, researcher)
        
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
                "keywords_researcher": "Webpage content analysis and keyword opportunity identification",
                "seo_analyst": "Strategic validation and implementation prioritisation"
            }
        }
    
    def execute_long_tail_analysis(self, page_content: PageContent, provider: AIProvider) -> Dict[str, str]:
        """Execute long-tail SEO analysis using CrewAI based on webpage content"""
        llm = self.model_manager.get_llm(provider)
        
        specialist = self.agent_factory.create_long_tail_specialist(llm)
        analyst = self.agent_factory.create_seo_analyst(llm)
        
        analysis_task = self.task_factory.create_long_tail_task(page_content, specialist)
        
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
                "long_tail_specialist": "Content-based long-tail keyword opportunity analysis",
                "seo_analyst": "Strategic validation and content alignment assessment"
            }
        }
    
    def execute_short_tail_strategy(self, page_content: PageContent, provider: AIProvider) -> Dict[str, str]:
        """Execute short-tail SEO strategy using CrewAI based on webpage content"""
        llm = self.model_manager.get_llm(provider)
        
        strategist = self.agent_factory.create_short_tail_strategist(llm)
        analyst = self.agent_factory.create_seo_analyst(llm)
        
        strategy_task = self.task_factory.create_short_tail_task(page_content, strategist)
        
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
                "short_tail_strategist": "Content authority assessment and competitive short-tail strategy",
                "seo_analyst": "Strategic validation and realistic goal setting"
            }
        }
    
    def execute_content_planning(self, page_content: PageContent, provider: AIProvider) -> Dict[str, str]:
        """Execute content planning using CrewAI based on webpage content"""
        llm = self.model_manager.get_llm(provider)
        
        planner = self.agent_factory.create_content_planner(llm)
        analyst = self.agent_factory.create_seo_analyst(llm)
        
        planning_task = self.task_factory.create_content_planning_task(page_content, planner)
        
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
                "content_planner": "Comprehensive content improvement and expansion strategy",
                "seo_analyst": "Strategic validation and implementation roadmap"
            }
        }
    
    def execute_comprehensive_analysis(self, page_content: PageContent, provider: AIProvider) -> Dict[str, str]:
        """Execute comprehensive SEO analysis using all agents"""
        llm = self.model_manager.get_llm(provider)
        
        # Create all specialist agents
        researcher = self.agent_factory.create_keywords_researcher(llm)
        long_tail_specialist = self.agent_factory.create_long_tail_specialist(llm)
        short_tail_strategist = self.agent_factory.create_short_tail_strategist(llm)
        content_planner = self.agent_factory.create_content_planner(llm)
        analyst = self.agent_factory.create_seo_analyst(llm)
        
        # Create comprehensive task
        comprehensive_task = self.task_factory.create_comprehensive_analysis_task(page_content, analyst)
        
        crew = Crew(
            agents=[researcher, long_tail_specialist, short_tail_strategist, content_planner, analyst],
            tasks=[comprehensive_task],
            verbose=True,
            process=Process.sequential
        )
        
        result = crew.kickoff()
        
        return {
            "primary_analysis": str(result),
            "agent_contributions": {
                "keywords_researcher": "Current keyword analysis and opportunity identification",
                "long_tail_specialist": "Long-tail keyword opportunity mapping",
                "short_tail_strategist": "Competitive short-tail keyword strategy",
                "content_planner": "Content improvement and expansion strategy",
                "seo_analyst": "Comprehensive analysis coordination and strategic validation"
            }
        }


class SEOURLAgent:
    """Main SEO Agent class for URL-based analysis using CrewAI"""
    
    def __init__(self):
        self.model_manager = AIModelManager()
        self.crew_manager = SEOCrewManager(self.model_manager)
        self.content_extractor = WebContentExtractor()
        self.results_history: List[SEOResult] = []
    
    def configure_api_keys(self, gpt4_key: Optional[str] = None, claude_key: Optional[str] = None):
        """Configure API keys for AI providers"""
        if gpt4_key:
            self.model_manager.configure_openai(gpt4_key)
        
        if claude_key:
            self.model_manager.configure_claude(claude_key)
    
    def execute_seo_task(self, task_type: SEOTaskType, url: str, provider: AIProvider) -> SEOResult:
        """Execute SEO task using CrewAI multi-agent system based on URL content"""
        print(f"Executing {task_type.value} for '{url}' using {provider.value}...")
        
        try:
            # Extract webpage content first
            page_content = self.content_extractor.extract_page_content(url)
            
            print("Initialising CrewAI agents for content analysis...")
            
            # Execute task based on type
            if task_type == SEOTaskType.KEYWORDS_RESEARCH:
                agent_outputs = self.crew_manager.execute_keywords_research(page_content, provider)
            elif task_type == SEOTaskType.LONG_TAIL_SEO:
                agent_outputs = self.crew_manager.execute_long_tail_analysis(page_content, provider)
            elif task_type == SEOTaskType.SHORT_TAIL_SEO:
                agent_outputs = self.crew_manager.execute_short_tail_strategy(page_content, provider)
            elif task_type == SEOTaskType.CONTENT_PLANNING:
                agent_outputs = self.crew_manager.execute_content_planning(page_content, provider)
            elif task_type == SEOTaskType.COMPREHENSIVE_ANALYSIS:
                agent_outputs = self.crew_manager.execute_comprehensive_analysis(page_content, provider)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")
            
            # Create timestamp for file naming
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Create result object
            result = SEOResult(
                task_type=task_type,
                url=url,
                page_content=page_content,
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
            print(f"✓ Results saved to url_analysis_{timestamp}.txt")
            return result
            
        except Exception as e:
            print(f"✗ Task failed: {str(e)}")
            raise
    
    def keywords_research(self, url: str, provider: AIProvider) -> SEOResult:
        """Perform keywords research analysis based on URL content"""
        return self.execute_seo_task(SEOTaskType.KEYWORDS_RESEARCH, url, provider)
    
    def long_tail_seo(self, url: str, provider: AIProvider) -> SEOResult:
        """Perform long-tail SEO analysis based on URL content"""
        return self.execute_seo_task(SEOTaskType.LONG_TAIL_SEO, url, provider)
    
    def short_tail_seo(self, url: str, provider: AIProvider) -> SEOResult:
        """Perform short-tail SEO analysis based on URL content"""
        return self.execute_seo_task(SEOTaskType.SHORT_TAIL_SEO, url, provider)
    
    def content_planning(self, url: str, provider: AIProvider) -> SEOResult:
        """Perform content planning analysis based on URL content"""
        return self.execute_seo_task(SEOTaskType.CONTENT_PLANNING, url, provider)
    
    def comprehensive_analysis(self, url: str, provider: AIProvider) -> SEOResult:
        """Perform comprehensive SEO analysis based on URL content"""
        return self.execute_seo_task(SEOTaskType.COMPREHENSIVE_ANALYSIS, url, provider)
    
    def _write_result_to_file(self, result: SEOResult, timestamp: str):
        """Write result to timestamped text file with page content summary"""
        filename = f"url_analysis_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # Write header
                f.write("="*80 + "\n")
                f.write("SEO URL ANALYSIS REPORT\n")
                f.write("="*80 + "\n\n")
                
                # Write metadata
                f.write(f"Task Type: {result.task_type.value.replace('_', ' ').title()}\n")
                f.write(f"Analysed URL: {result.url}\n")
                f.write(f"AI Provider: {result.provider.value}\n")
                f.write(f"Analysis Date: {result.timestamp}\n")
                f.write(f"Content Extraction: {result.page_content.extraction_timestamp}\n")
                f.write(f"Report File: {filename}\n\n")
                
                # Write page content summary
                f.write("PAGE CONTENT SUMMARY\n")
                f.write("-"*40 + "\n")
                f.write(f"Title: {result.page_content.title}\n")
                f.write(f"Meta Description: {result.page_content.meta_description}\n")
                f.write(f"Word Count: {result.page_content.word_count}\n")
                f.write(f"Images: {len(result.page_content.images)}\n")
                f.write(f"Internal Links: {len(result.page_content.links['internal'])}\n")
                f.write(f"External Links: {len(result.page_content.links['external'])}\n\n")
                
                # Write headings structure
                f.write("CONTENT STRUCTURE\n")
                f.write("-"*40 + "\n")
                for level, headings in result.page_content.headings.items():
                    if headings:
                        f.write(f"{level.upper()}: {len(headings)} headings\n")
                        for heading in headings[:5]:  # Show first 5 headings per level
                            f.write(f"  • {heading}\n")
                        if len(headings) > 5:
                            f.write(f"  ... and {len(headings) - 5} more\n")
                f.write("\n")
                
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
                
                # Write extracted content preview
                f.write("EXTRACTED CONTENT PREVIEW\n")
                f.write("-"*40 + "\n")
                f.write("First 2000 characters of extracted content:\n\n")
                f.write(result.page_content.content_markdown[:2000])
                f.write("\n\n")
                
                # Write footer
                f.write("="*80 + "\n")
                f.write("End of SEO URL Analysis Report\n")
                f.write(f"Generated by Finn SEO URL Agent on {result.timestamp}\n")
                f.write("="*80 + "\n")
                
        except Exception as e:
            print(f"Warning: Failed to write result to file {filename}: {str(e)}")
    
    def export_results(self, filename: str = "seo_url_results.json"):
        """Export all results to JSON file"""
        if not self.results_history:
            print("No results to export")
            return
        
        export_data = []
        for result in self.results_history:
            export_data.append({
                "task_type": result.task_type.value,
                "url": result.url,
                "page_content": {
                    "title": result.page_content.title,
                    "meta_description": result.page_content.meta_description,
                    "word_count": result.page_content.word_count,
                    "headings": result.page_content.headings,
                    "extraction_timestamp": result.page_content.extraction_timestamp
                },
                "results": result.results,
                "provider": result.provider.value,
                "timestamp": result.timestamp,
                "agent_contributions": result.agent_outputs
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results exported to {filename}")
    
    def display_result(self, result: SEOResult):
        """Display formatted result with page content and agent contributions"""
        print(f"\n{'='*80}")
        print(f"Task: {result.task_type.value.replace('_', ' ').title()}")
        print(f"URL: {result.url}")
        print(f"Provider: {result.provider.value}")
        print(f"Analysis Timestamp: {result.timestamp}")
        print(f"{'='*80}")
        
        # Display page content summary
        print("Page Content Summary:")
        print(f"  Title: {result.page_content.title}")
        print(f"  Meta Description: {result.page_content.meta_description[:100]}...")
        print(f"  Word Count: {result.page_content.word_count}")
        print(f"  Images: {len(result.page_content.images)}")
        print(f"  Links: {len(result.page_content.links['internal'])} internal, {len(result.page_content.links['external'])} external")
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


class SEOURLAgentCLI:
    """Command-line interface for the URL-based SEO Agent"""
    
    def __init__(self):
        self.agent = SEOURLAgent()
        self.setup_complete = False
    
    def setup_api_keys(self):
        """Interactive API key setup"""
        print("Finn SEO URL Agent Setup")
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
        print("\nFinn SEO URL Agent - Main Menu")
        print("=" * 50)
        print("1. Keywords Research (URL-based)")
        print("2. Long-tail SEO (URL-based)")
        print("3. Short-tail SEO (URL-based)")
        print("4. Content Planning (URL-based)")
        print("5. Comprehensive Analysis (All agents)")
        print("6. Export Results (JSON)")
        print("7. View Results History")
        print("8. List Result Files")
        print("9. Exit")
        print("=" * 50)
    
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
    
    def get_url_input(self) -> str:
        """Get and validate URL input from user"""
        while True:
            url = input("Enter URL to analyse: ").strip()
            if not url:
                print("URL cannot be empty")
                continue
            
            # Add https:// if no protocol specified
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Basic URL validation
            try:
                parsed = urllib.parse.urlparse(url)
                if not parsed.netloc:
                    print("Invalid URL format. Please try again.")
                    continue
                return url
            except Exception:
                print("Invalid URL format. Please try again.")
                continue
    
    def run_task(self, task_type: SEOTaskType):
        """Run a specific SEO task with multi-agent system"""
        url = self.get_url_input()
        provider = self.get_provider_choice()
        
        print("\nInitialising URL content extraction and CrewAI multi-agent system...")
        print(f"Task: {task_type.value.replace('_', ' ').title()}")
        print(f"URL: {url}")
        print(f"Provider: {provider.value}")
        print("This may take a few minutes as content is extracted and multiple agents collaborate...")
        
        try:
            result = self.agent.execute_seo_task(task_type, url, provider)
            self.agent.display_result(result)
        except Exception as e:
            print(f"Error executing task: {str(e)}")
    
    def list_result_files(self):
        """List all URL analysis result files in current directory"""
        result_files = [f for f in os.listdir('.') if f.startswith('url_analysis_') and f.endswith('.txt')]
        
        if not result_files:
            print("No URL analysis result files found in current directory")
            return
        
        result_files.sort(reverse=True)  # Sort by newest first
        
        print(f"\nFound {len(result_files)} URL analysis result files:")
        print("=" * 60)
        
        for i, filename in enumerate(result_files, 1):
            # Extract timestamp from filename
            timestamp_part = filename.replace('url_analysis_', '').replace('.txt', '')
            try:
                # Convert timestamp to readable format
                timestamp_obj = time.strptime(timestamp_part, "%Y%m%d_%H%M%S")
                readable_time = time.strftime("%Y-%m-%d %H:%M:%S", timestamp_obj)
                file_size = os.path.getsize(filename)
                print(f"{i:2d}. {filename} ({readable_time}) - {file_size} bytes")
            except ValueError:
                print(f"{i:2d}. {filename} (Invalid timestamp format)")
        
        print("=" * 60)
    
    def view_history(self):
        """Display results history"""
        if not self.agent.results_history:
            print("No results in history")
            return
        
        print(f"\nURL Analysis Results History ({len(self.agent.results_history)} items)")
        print("=" * 80)
        
        for i, result in enumerate(self.agent.results_history, 1):
            agents_count = len(result.agent_outputs)
            url_short = result.url[:50] + "..." if len(result.url) > 50 else result.url
            print(f"{i}. {result.task_type.value} - {url_short} ({result.provider.value}) - {agents_count} agents - {result.timestamp}")
        
        try:
            choice = int(input("\nEnter result number to view (0 to go back): "))
            if 1 <= choice <= len(self.agent.results_history):
                self.agent.display_result(self.agent.results_history[choice - 1])
        except ValueError:
            pass
    
    def run(self):
        """Run the CLI application"""
        print("Welcome to Finn SEO URL Agent")
        print("Multi-Agent URL-based SEO Analysis System")
        print("=" * 60)
        print("This tool extracts webpage content and performs comprehensive SEO analysis")
        print("using multiple AI agents working together.")
        print("=" * 60)
        
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
                    self.run_task(SEOTaskType.COMPREHENSIVE_ANALYSIS)
                elif choice == 6:
                    filename = input("Export filename (default: seo_url_results.json): ").strip()
                    self.agent.export_results(filename if filename else "seo_url_results.json")
                elif choice == 7:
                    self.view_history()
                elif choice == 8:
                    self.list_result_files()
                elif choice == 9:
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
        cli = SEOURLAgentCLI()
        cli.run()
    except Exception as e:
        print(f"Application error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
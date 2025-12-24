"""
Mock News Provider - Realistic mock news data for testing.

Provides pre-defined news articles for common tickers.
Supports testing without NewsAPI.org API calls.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from src.application.ports.news_port import NewsPort
from src.config.logging import get_logger
from src.domain.models import NewsArticle

logger = get_logger(__name__)


def _now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def _get_mock_news_data() -> Dict[str, List[Dict]]:
    """Generate mock news data with current timestamps."""
    now = _now()
    return {
        "TSLA": [
            {
                "title": "Tesla Reports Record Q3 Deliveries, Beats Wall Street Expectations",
                "description": "Electric vehicle maker Tesla delivered 435,000 vehicles in Q3 2024, surpassing analyst estimates and setting a new company record.",
                "url": "https://example.com/news/tesla-q3-deliveries-record",
                "source": "Financial Times",
                "published_at": now - timedelta(days=2),
                "content": "Tesla Inc. reported record vehicle deliveries for the third quarter of 2024, surpassing Wall Street expectations and demonstrating continued strong demand for electric vehicles despite broader economic concerns. The company delivered 435,059 vehicles during the quarter, beating analyst estimates of 420,000 units.",
            },
            {
                "title": "Tesla Expands Gigafactory Texas Production Capacity",
                "description": "Tesla announces $3.6 billion expansion of its Austin Gigafactory, adding new production lines for Cybertruck and next-gen vehicles.",
                "url": "https://example.com/news/tesla-gigafactory-expansion",
                "source": "Reuters",
                "published_at": now - timedelta(days=5),
                "content": "Tesla is expanding its Gigafactory in Austin, Texas with a $3.6 billion investment that will add multiple new production lines. The expansion will support increased Cybertruck production and prepare facilities for next-generation vehicle manufacturing.",
            },
            {
                "title": "Analysts Raise Tesla Price Targets Following Strong Performance",
                "description": "Multiple Wall Street firms upgrade Tesla stock following better-than-expected delivery numbers and margin improvements.",
                "url": "https://example.com/news/tesla-analyst-upgrades",
                "source": "Bloomberg",
                "published_at": now - timedelta(days=7),
                "content": "Several major Wall Street analysts have raised their price targets for Tesla stock following the company's strong Q3 performance. Morgan Stanley increased its target to $310 from $280, citing improved production efficiency and strong demand.",
            },
            {
                "title": "Tesla Faces Increased Competition in China EV Market",
                "description": "Chinese EV makers BYD and NIO report strong sales, intensifying competition in Tesla's second-largest market.",
                "url": "https://example.com/news/tesla-china-competition",
                "source": "Wall Street Journal",
                "published_at": now - timedelta(days=10),
                "content": "Tesla faces mounting competition in China as domestic EV manufacturers report record sales. BYD, now the world's largest EV producer by volume, continues to gain market share with competitive pricing and advanced technology.",
            },
            {
                "title": "Tesla Energy Storage Revenue Surges 52% Year-Over-Year",
                "description": "Tesla's energy storage business shows strong growth, becoming an increasingly important revenue driver for the company.",
                "url": "https://example.com/news/tesla-energy-storage-growth",
                "source": "CNBC",
                "published_at": now - timedelta(days=14),
                "content": "Tesla's energy storage division reported 52% year-over-year revenue growth, highlighting the company's successful diversification beyond electric vehicles. The Megapack and Powerwall products continue to see strong demand globally.",
            },
        ],
        "AAPL": [
            {
                "title": "Apple Unveils iPhone 16 with Advanced AI Features",
                "description": "Apple introduces next-generation iPhone with integrated Apple Intelligence features and improved performance.",
                "url": "https://example.com/news/apple-iphone-16-launch",
                "source": "The Verge",
                "published_at": now - timedelta(days=3),
                "content": "Apple unveiled the iPhone 16 series featuring Apple Intelligence, the company's suite of AI-powered features. The new devices include enhanced Siri capabilities, improved photo editing, and advanced computational photography.",
            },
            {
                "title": "Apple Services Revenue Hits New All-Time High",
                "description": "App Store, Apple Music, and iCloud drive services segment to record quarterly revenue.",
                "url": "https://example.com/news/apple-services-record",
                "source": "MacRumors",
                "published_at": now - timedelta(days=8),
                "content": "Apple's Services segment achieved record revenue of $24.2 billion in the quarter, representing 14% year-over-year growth. The segment, which includes App Store, Apple Music, iCloud, and Apple TV+, continues to be a key growth driver.",
            },
            {
                "title": "Apple Invests $1B in Vietnam Manufacturing Expansion",
                "description": "Tech giant expands supply chain diversification with major investment in Vietnamese manufacturing facilities.",
                "url": "https://example.com/news/apple-vietnam-investment",
                "source": "Financial Times",
                "published_at": now - timedelta(days=12),
                "content": "Apple is investing $1 billion to expand its manufacturing presence in Vietnam as part of its ongoing supply chain diversification strategy. The investment will support production of AirPods, iPads, and other devices.",
            },
            {
                "title": "EU Antitrust Ruling Impacts Apple App Store Policies",
                "description": "European Commission ruling requires Apple to allow alternative payment systems in App Store.",
                "url": "https://example.com/news/apple-eu-antitrust",
                "source": "Reuters",
                "published_at": now - timedelta(days=18),
                "content": "Apple is adjusting its App Store policies in Europe following regulatory pressure from the European Commission. The company will allow developers to use alternative payment systems, potentially impacting its services revenue.",
            },
            {
                "title": "Apple Vision Pro Sees Growing Enterprise Adoption",
                "description": "Major companies adopt Vision Pro for training, design, and collaboration applications.",
                "url": "https://example.com/news/apple-vision-pro-enterprise",
                "source": "Bloomberg",
                "published_at": now - timedelta(days=22),
                "content": "Apple's Vision Pro headset is gaining traction in enterprise environments, with companies like Porsche, SAP, and Walmart deploying the device for employee training and product design applications.",
            },
        ],
        "GOOGL": [
            {
                "title": "Google Cloud Revenue Grows 28% as AI Services Gain Traction",
                "description": "Alphabet's cloud division sees strong growth driven by AI and machine learning services demand.",
                "url": "https://example.com/news/google-cloud-growth",
                "source": "TechCrunch",
                "published_at": now - timedelta(days=4),
                "content": "Google Cloud reported 28% year-over-year revenue growth, driven by strong demand for AI and machine learning services. The division generated $11.4 billion in quarterly revenue, narrowing the gap with market leaders AWS and Azure.",
            },
            {
                "title": "Alphabet Announces $70 Billion Stock Buyback Program",
                "description": "Google parent company expands shareholder returns with massive new share repurchase authorization.",
                "url": "https://example.com/news/alphabet-buyback",
                "source": "Wall Street Journal",
                "published_at": now - timedelta(days=9),
                "content": "Alphabet Inc. announced a $70 billion stock buyback program, its largest ever, signaling confidence in the company's financial strength and commitment to returning value to shareholders.",
            },
            {
                "title": "Google DeepMind Achieves Breakthrough in Protein Structure Prediction",
                "description": "AI research lab's latest model can predict protein structures with unprecedented accuracy.",
                "url": "https://example.com/news/deepmind-protein-breakthrough",
                "source": "Nature",
                "published_at": now - timedelta(days=15),
                "content": "Google DeepMind's latest AlphaFold model has achieved a major breakthrough in protein structure prediction, with implications for drug discovery and understanding biological processes. The advancement represents a significant leap in computational biology.",
            },
            {
                "title": "DOJ Antitrust Case Against Google Enters Final Arguments",
                "description": "US Department of Justice concludes its landmark antitrust trial against Google's search dominance.",
                "url": "https://example.com/news/google-antitrust-trial",
                "source": "New York Times",
                "published_at": now - timedelta(days=20),
                "content": "The US Department of Justice has concluded its antitrust case against Google, arguing that the company illegally maintained its monopoly in search through exclusive agreements with device manufacturers and browsers.",
            },
            {
                "title": "YouTube Shorts Monetization Expands to More Creators",
                "description": "Google expands revenue sharing program for short-form video creators on YouTube.",
                "url": "https://example.com/news/youtube-shorts-monetization",
                "source": "Variety",
                "published_at": now - timedelta(days=25),
                "content": "YouTube is expanding its Shorts monetization program, allowing more creators to earn revenue from short-form video content. The move is aimed at competing more effectively with TikTok and Instagram Reels.",
            },
        ],
        "MSFT": [
            {
                "title": "Microsoft Azure Revenue Jumps 29% on AI Demand",
                "description": "Cloud computing giant sees accelerated growth as enterprises adopt AI services.",
                "url": "https://example.com/news/azure-ai-growth",
                "source": "Bloomberg",
                "published_at": now - timedelta(days=2),
                "content": "Microsoft's Azure cloud platform reported 29% revenue growth, accelerating from previous quarters as enterprise customers adopt AI and machine learning capabilities. The company's partnership with OpenAI continues to drive demand for its AI services.",
            },
            {
                "title": "Microsoft Copilot Reaches 1 Million Enterprise Users",
                "description": "AI assistant for Microsoft 365 sees rapid adoption among business customers.",
                "url": "https://example.com/news/microsoft-copilot-adoption",
                "source": "The Information",
                "published_at": now - timedelta(days=7),
                "content": "Microsoft's AI-powered Copilot for Microsoft 365 has reached 1 million enterprise users, demonstrating strong demand for AI productivity tools in the workplace. Major companies including Accenture, EY, and Walmart have deployed the service.",
            },
            {
                "title": "Microsoft Gaming Revenue Surges Following Activision Acquisition",
                "description": "Gaming segment posts strong results with contributions from newly acquired Activision Blizzard.",
                "url": "https://example.com/news/microsoft-gaming-growth",
                "source": "IGN",
                "published_at": now - timedelta(days=12),
                "content": "Microsoft's gaming division reported significant revenue growth following the completion of its Activision Blizzard acquisition. Popular franchises including Call of Duty and World of Warcraft are now contributing to Xbox's game catalog.",
            },
            {
                "title": "Microsoft Announces Plans for Nuclear-Powered Data Centers",
                "description": "Tech giant explores small modular reactors to power AI infrastructure sustainably.",
                "url": "https://example.com/news/microsoft-nuclear-datacenters",
                "source": "Reuters",
                "published_at": now - timedelta(days=16),
                "content": "Microsoft is exploring the use of small modular nuclear reactors to power its data centers, addressing the growing energy demands of AI workloads while meeting sustainability goals. The company has signed agreements with nuclear energy startups.",
            },
            {
                "title": "LinkedIn Reaches 1 Billion Members Milestone",
                "description": "Microsoft-owned professional network continues global growth trajectory.",
                "url": "https://example.com/news/linkedin-billion-members",
                "source": "TechCrunch",
                "published_at": now - timedelta(days=21),
                "content": "LinkedIn has reached 1 billion members globally, with strong growth in Asia-Pacific and emerging markets. The professional networking platform, owned by Microsoft, continues to expand its offerings in learning, recruiting, and content creation.",
            },
        ],
    }


def _get_default_news() -> List[Dict]:
    """Generate default news data with current timestamps."""
    now = _now()
    return [
        {
            "title": "Market Update: Technology Sector Shows Mixed Performance",
            "description": "Tech stocks show varied performance amid economic uncertainty and interest rate speculation.",
            "url": "https://example.com/news/market-update",
            "source": "Market Watch",
            "published_at": now - timedelta(days=1),
            "content": "The technology sector showed mixed performance today as investors weighed economic data against corporate earnings reports. Major indices saw moderate volatility as the market digested the latest Federal Reserve commentary.",
        },
        {
            "title": "Analysts Provide Outlook on Technology Investments",
            "description": "Investment analysts share perspectives on tech sector opportunities and risks.",
            "url": "https://example.com/news/tech-outlook",
            "source": "Bloomberg",
            "published_at": now - timedelta(days=5),
            "content": "Wall Street analysts have provided updated outlooks on the technology sector, highlighting opportunities in AI and cloud computing while noting risks from regulatory scrutiny and valuation concerns.",
        },
        {
            "title": "Q3 Earnings Season Approaches for Tech Companies",
            "description": "Investors prepare for upcoming quarterly reports from major technology firms.",
            "url": "https://example.com/news/earnings-preview",
            "source": "CNBC",
            "published_at": now - timedelta(days=10),
            "content": "Technology companies are preparing to report third-quarter earnings, with investors focused on revenue growth, profitability metrics, and forward guidance. AI investments and cloud growth remain key themes.",
        },
        {
            "title": "Global Supply Chain Conditions Improve for Tech Sector",
            "description": "Semiconductor and electronics supply chains show signs of normalization.",
            "url": "https://example.com/news/supply-chain",
            "source": "Financial Times",
            "published_at": now - timedelta(days=15),
            "content": "Global supply chain conditions for the technology sector continue to improve, with semiconductor lead times shortening and component availability increasing. Industry executives report more stable procurement conditions.",
        },
        {
            "title": "Technology Sector Wages Remain Competitive Despite Layoffs",
            "description": "Tech industry maintains competitive compensation despite workforce adjustments.",
            "url": "https://example.com/news/tech-wages",
            "source": "Wall Street Journal",
            "published_at": now - timedelta(days=20),
            "content": "Despite recent layoffs at several major technology companies, wages in the sector remain competitive as demand for specialized skills in AI, cybersecurity, and cloud computing continues to outpace supply.",
        },
    ]


class MockNews(NewsPort):
    """
    Mock news provider with realistic pre-defined articles.

    Provides consistent, realistic news data for testing
    without requiring NewsAPI.org API calls.
    """

    def __init__(self):
        self._call_count = 0

    async def search_news(
        self,
        query: str,
        num_articles: int = 5,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        language: str = "en",
    ) -> List[NewsArticle]:
        """Search for news articles (mock implementation)."""
        self._call_count += 1

        logger.debug(
            "mock_news_search",
            query=query,
            num_articles=num_articles,
            call_count=self._call_count,
        )

        # Try to find ticker in query
        ticker = self._extract_ticker(query)
        mock_data = _get_mock_news_data()
        articles_data = mock_data.get(ticker, _get_default_news())

        # Apply date filters if provided
        filtered = articles_data
        if from_date:
            filtered = [a for a in filtered if a["published_at"] >= from_date]
        if to_date:
            filtered = [a for a in filtered if a["published_at"] <= to_date]

        # Limit results
        filtered = filtered[:num_articles]

        return [NewsArticle(**article) for article in filtered]

    async def get_company_news(
        self,
        company_name: str,
        ticker: str,
        num_articles: int = 5,
    ) -> List[NewsArticle]:
        """Get news for a specific company (mock implementation)."""
        self._call_count += 1

        logger.debug(
            "mock_news_company",
            company_name=company_name,
            ticker=ticker,
            num_articles=num_articles,
            call_count=self._call_count,
        )

        # Get articles for ticker
        ticker_upper = ticker.upper()
        mock_data = _get_mock_news_data()
        articles_data = mock_data.get(ticker_upper, _get_default_news())

        # Limit results
        limited = articles_data[:num_articles]

        return [NewsArticle(**article) for article in limited]

    async def health_check(self) -> bool:
        """Mock is always healthy."""
        return True

    def _extract_ticker(self, query: str) -> str:
        """Extract ticker symbol from query."""
        query_upper = query.upper()

        # Check for known tickers
        known_tickers = ["TSLA", "AAPL", "GOOGL", "MSFT"]
        for ticker in known_tickers:
            if ticker in query_upper:
                return ticker

        return ""

    def get_call_count(self) -> int:
        """Get the number of calls made."""
        return self._call_count

    def reset(self) -> None:
        """Reset the mock state."""
        self._call_count = 0

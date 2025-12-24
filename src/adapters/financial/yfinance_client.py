"""
YFinance Client - Real financial data from Yahoo Finance via yfinance.

This module implements the Financial port using the yfinance library
for fetching real stock data and company financials.
"""

import asyncio
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf

from src.application.ports.financial_port import FinancialPort
from src.config.logging import get_logger
from src.domain.exceptions import FinancialDataError
from src.domain.models import FinancialData, QuarterlyRevenue

logger = get_logger(__name__)


class YFinanceClient(FinancialPort):
    """
    Yahoo Finance client for fetching real financial data.

    Uses the yfinance library to fetch stock prices, market data,
    and company financials from Yahoo Finance.

    Note: yfinance is a synchronous library, so we use asyncio.to_thread
    to run operations in a thread pool to avoid blocking the event loop.
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize the YFinance client.

        Args:
            timeout: Request timeout in seconds
        """
        self._timeout = timeout

    async def get_company_data(self, ticker: str) -> FinancialData:
        """Get comprehensive financial data for a company."""
        ticker_upper = ticker.upper()

        logger.debug("yfinance_get_company_data", ticker=ticker_upper)

        try:
            # Run yfinance in thread pool since it's synchronous
            data = await asyncio.to_thread(self._fetch_company_data, ticker_upper)

            logger.info(
                "yfinance_company_data_fetched",
                ticker=ticker_upper,
                price=data.current_price,
            )

            return data

        except Exception as e:
            logger.error(
                "yfinance_get_company_data_error",
                ticker=ticker_upper,
                error=str(e),
            )
            raise FinancialDataError(
                message=f"Failed to fetch financial data for {ticker_upper}: {e}",
                ticker=ticker_upper,
            )

    def _fetch_company_data(self, ticker: str) -> FinancialData:
        """Synchronous helper to fetch company data from yfinance."""
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get company name
        company_name = info.get("longName") or info.get("shortName") or f"{ticker} Inc."

        # Get current price - try multiple fields
        current_price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )

        # Get price change percentage
        price_change_percent = info.get("regularMarketChangePercent")

        # Get market cap
        market_cap = info.get("marketCap")

        # Get P/E ratio (trailing)
        pe_ratio = info.get("trailingPE")

        # Get 52-week high/low
        fifty_two_week_high = info.get("fiftyTwoWeekHigh")
        fifty_two_week_low = info.get("fiftyTwoWeekLow")

        # Get quarterly revenue data (now 4 quarters)
        quarterly_revenue = self._get_quarterly_revenue(stock, num_quarters=4)

        # Profitability metrics
        gross_margin = info.get("grossMargins")
        operating_margin = info.get("operatingMargins")
        profit_margin = info.get("profitMargins")

        # Balance sheet health
        total_debt = info.get("totalDebt")
        total_cash = info.get("totalCash")
        debt_to_equity = info.get("debtToEquity")
        current_ratio = info.get("currentRatio")

        # Growth metrics
        revenue_growth = info.get("revenueGrowth")
        earnings_growth = info.get("earningsGrowth")

        # Valuation metrics
        forward_pe = info.get("forwardPE")
        peg_ratio = info.get("pegRatio")
        price_to_book = info.get("priceToBook")
        price_to_sales = info.get("priceToSalesTrailing12Months")
        enterprise_value = info.get("enterpriseValue")
        ev_to_ebitda = info.get("enterpriseToEbitda")

        # Analyst data
        analyst_target_mean = info.get("targetMeanPrice")
        analyst_target_low = info.get("targetLowPrice")
        analyst_target_high = info.get("targetHighPrice")
        analyst_recommendation = info.get("recommendationKey")
        number_of_analysts = info.get("numberOfAnalystOpinions")

        # Dividend info
        dividend_yield = info.get("dividendYield")
        payout_ratio = info.get("payoutRatio")

        # Additional context
        sector = info.get("sector")
        industry = info.get("industry")
        beta = info.get("beta")
        short_ratio = info.get("shortRatio")

        return FinancialData(
            ticker=ticker,
            company_name=company_name,
            current_price=current_price,
            price_change_percent=price_change_percent,
            market_cap=market_cap,
            pe_ratio=pe_ratio,
            quarterly_revenue=quarterly_revenue,
            fifty_two_week_high=fifty_two_week_high,
            fifty_two_week_low=fifty_two_week_low,
            data_timestamp=datetime.now(timezone.utc),
            # Profitability
            gross_margin=gross_margin,
            operating_margin=operating_margin,
            profit_margin=profit_margin,
            # Balance sheet
            total_debt=total_debt,
            total_cash=total_cash,
            debt_to_equity=debt_to_equity,
            current_ratio=current_ratio,
            # Growth
            revenue_growth=revenue_growth,
            earnings_growth=earnings_growth,
            # Valuation
            forward_pe=forward_pe,
            peg_ratio=peg_ratio,
            price_to_book=price_to_book,
            price_to_sales=price_to_sales,
            enterprise_value=enterprise_value,
            ev_to_ebitda=ev_to_ebitda,
            # Analyst data
            analyst_target_mean=analyst_target_mean,
            analyst_target_low=analyst_target_low,
            analyst_target_high=analyst_target_high,
            analyst_recommendation=analyst_recommendation,
            number_of_analysts=number_of_analysts,
            # Dividends
            dividend_yield=dividend_yield,
            payout_ratio=payout_ratio,
            # Context
            sector=sector,
            industry=industry,
            beta=beta,
            short_ratio=short_ratio,
        )

    def _get_quarterly_revenue(self, stock: yf.Ticker, num_quarters: int = 4) -> list[QuarterlyRevenue]:
        """Extract quarterly revenue data from yfinance Ticker."""
        quarterly_revenue = []

        try:
            # Get quarterly financials
            quarterly_financials = stock.quarterly_financials

            if quarterly_financials is None or quarterly_financials.empty:
                return quarterly_revenue

            # Look for revenue row - try multiple common names
            revenue_row = None
            for row_name in ["Total Revenue", "Revenue", "Operating Revenue"]:
                if row_name in quarterly_financials.index:
                    revenue_row = quarterly_financials.loc[row_name]
                    break

            if revenue_row is None:
                return quarterly_revenue

            # Get the requested number of quarters
            columns = list(revenue_row.index)[:num_quarters]

            for i, col in enumerate(columns):
                try:
                    revenue_value = revenue_row[col]
                    if revenue_value is None or (hasattr(revenue_value, "__float__") and revenue_value != revenue_value):
                        continue

                    # Format quarter string from the column date
                    if hasattr(col, "strftime"):
                        quarter_num = (col.month - 1) // 3 + 1
                        quarter_str = f"Q{quarter_num} {col.year}"
                    else:
                        quarter_str = f"Q{i + 1}"

                    # Calculate year-over-year change if we have enough data
                    yoy_change = None
                    if len(columns) > i + 4:
                        try:
                            prev_year_revenue = revenue_row[columns[i + 4]]
                            if prev_year_revenue and prev_year_revenue > 0:
                                yoy_change = ((float(revenue_value) - float(prev_year_revenue)) / float(prev_year_revenue)) * 100
                        except (KeyError, IndexError, TypeError, ValueError):
                            pass

                    quarterly_revenue.append(
                        QuarterlyRevenue(
                            quarter=quarter_str,
                            revenue=float(revenue_value),
                            year_over_year_change=yoy_change,
                        )
                    )
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning(
                        "yfinance_quarterly_revenue_parse_error",
                        column=str(col),
                        error=str(e),
                    )
                    continue

        except Exception as e:
            logger.warning(
                "yfinance_quarterly_revenue_error",
                error=str(e),
            )

        return quarterly_revenue

    async def get_stock_price(self, ticker: str) -> Optional[float]:
        """Get current stock price."""
        ticker_upper = ticker.upper()

        logger.debug("yfinance_get_stock_price", ticker=ticker_upper)

        try:
            price = await asyncio.to_thread(self._fetch_stock_price, ticker_upper)

            logger.info(
                "yfinance_stock_price_fetched",
                ticker=ticker_upper,
                price=price,
            )

            return price

        except Exception as e:
            logger.error(
                "yfinance_get_stock_price_error",
                ticker=ticker_upper,
                error=str(e),
            )
            return None

    def _fetch_stock_price(self, ticker: str) -> Optional[float]:
        """Synchronous helper to fetch stock price from yfinance."""
        stock = yf.Ticker(ticker)
        info = stock.info

        # Try multiple price fields
        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )

        return float(price) if price else None

    async def get_company_name(self, ticker: str) -> Optional[str]:
        """Get company name from ticker."""
        ticker_upper = ticker.upper()

        logger.debug("yfinance_get_company_name", ticker=ticker_upper)

        try:
            name = await asyncio.to_thread(self._fetch_company_name, ticker_upper)

            logger.info(
                "yfinance_company_name_fetched",
                ticker=ticker_upper,
                name=name,
            )

            return name

        except Exception as e:
            logger.error(
                "yfinance_get_company_name_error",
                ticker=ticker_upper,
                error=str(e),
            )
            return None

    def _fetch_company_name(self, ticker: str) -> Optional[str]:
        """Synchronous helper to fetch company name from yfinance."""
        stock = yf.Ticker(ticker)
        info = stock.info

        return info.get("longName") or info.get("shortName")

    async def health_check(self) -> bool:
        """Check if Yahoo Finance is available."""
        try:
            # Try to fetch data for a well-known ticker
            price = await self.get_stock_price("AAPL")
            return price is not None
        except Exception as e:
            logger.error("yfinance_health_check_failed", error=str(e))
            return False

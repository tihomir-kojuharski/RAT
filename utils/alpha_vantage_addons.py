from alpha_vantage.alphavantage import AlphaVantage as av
from alpha_vantage.fundamentaldata import FundamentalData


class FundamentalDataWithEarnings(FundamentalData):
    """This class implements all the api calls to fundamental data
    """

    def __init__(self, *args, **kwargs):
        super(FundamentalDataWithEarnings, self).__init__(*args, **kwargs)

    @av._output_format
    @av._call_api_on_func
    def get_earnings(self, symbol):
        """
        Returns the annual and quarterly balance sheets for the company of interest.
        Data is generally refreshed on the same day a company reports its latest
        earnings and financials.

        Keyword Arguments:
            symbol:  the symbol for the equity we want to get its data
        """
        _FUNCTION_KEY = 'EARNINGS'
        return _FUNCTION_KEY, None, None

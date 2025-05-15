# Data sources

## Wind generation

Wind generation data is obtained from [renewables.ninja](https://renewables.ninja/), for years 2010-2019.

It is assumed that wind power is taken from a future wind farm located in the [IJmuiden Ver](https://offshorewind.rvo.nl/page/view/5c06ac88-c12f-4903-89f3-27d66937b7e9/general-information-ijmuiden-ver) wind farm development zone in the North Sea [(52.7,3.5)](https://map.4coffshore.com/offshorewind/), the largest development zone located close to Rotterdam.

The power model for Vestas V164 9500 wind turbines is used to provide the best representation of large-scale offshore wind generation.

## Solar generation

Solar generation data is obtained from [EU JRC](https://re.jrc.ec.europa.eu/pvg_tools/en/), for years 2010-2019.

Solar generation is assumed to be on-site, in the [assumed industrial site](https://www.h2-fifty.com/) in the Port of Rotterdam (51.95,4.1).

## Grid electricity price

Day-ahead electricity prices for the Netherlands are taken from [entso-e](https://transparency.entsoe.eu/transmission-domain/r2/dayAheadPrices/show?areaType=BZN). Note realtime pricing is more commercially sensitive and so is expensive to access and not publicly sharable.

2023 prices are used as they are considered to most reflective of future electricity price trends, e.g. containing periods of negative prices, and not being distorted by: 2022 gas shortages, COVID effects, etc.

From the [Dutch statistics body](https://www.cbs.nl/en-gb/figures/detail/85592ENG), in 2023 the average consumer electricity price was around €330/MWh. So, the day-ahead prices are scaled to an average of €300/MWh to reflect commercial prices.

2024 prices, and 2023 prices normalised to €270/MWh and €330/MWh (plus-minus 10%), are used for sensitivity analysis.

## Grid electricity carbon intensity

2023 grid electricity carbon intensity data for the Netherlands is taken from [electricitymaps.com](https://portal.electricitymaps.com/datasets/NL).

2024 data is used for sensitivity analysis.
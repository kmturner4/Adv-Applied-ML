
# Predicting Housing Prices in Northern Virginia: a Time Series Regression with LSTM weight tuning
By: Kaitlyn Turner

## Abstract

Northern Virginia is an expensive housing market that has seen continued growth in the past 20 years, with the 2024 year-to-date median sale price of $813,724 (NVAR 2024). I use housing data from the past 24 years from the Northern Virginia Association of Realtors to create a time series regression model. I employ a neural network based on long-short term memory (LSTM) units to select the regression weights. The model with LSTM weight tuning vastly outperforms a multiple regression model. 

## Introduction

The housing in the Northern Virginia is the most expensive in the state, with high demand and moderately restricted supply. The proximity to D.C., in addition to many government, contracting jobs, defense, and financial jobs, makes the adjacent counties of Virginia valuable. The majority of the housing is detached single family homes. The median housing price in March 2024 was $631,564 (Long and Foster 2024). For the counties that are in closer proximity to D.C. and are in the data series I use, the median housing price was $730,000 (NVAR 2024). 

With the exception of the 2007 housing crash, the price of housing in Northern Virginia has been steadily on the rise since 2000. This trend has not eased in recent years, either, with the year over year rates for inventory down 22%, housing listings down 18%, days on the market down 20%, and sales price up 10% since March of 2023 (NVAR 2024). All of these characterize the situation of undersupplied housing, which contributes to the expensive and rising housing prices for the area. 

In my research, I found no examples using a long short term memory (LSTM) neural network in the context of Northern Virginia housing prices. There are a few published of machine learning methods to forecast real estate prices in some counties in the region. Two studies found that random forest models used for real estate forecasting in Arlington County (Wang and Wu, 2018) and Fairfax County (Hu et al., 2022) were superior to other methods. 

Other machine learning techniques have been applied to real estate price forecasting. Wang et al. tested against grid search algorithms and other optimization methods alongside support vector machines (SVM) and found that that particle swarm optimization in conjunction with SVM forecasted real estate prices best (2014). Another similar study of real estate prices in Saudi Arabia found that random forests outperformed regression and decision tree models (Louati et al. 2021). 

LSTM has been applied to forecasting in other financial domains. Fisher and Krauss tested LSTM networks as well as random forests, logistic regression models, and other convolutional neural networks and found that LSTM networks were superior for forecasting the S&P 500 (2018). Another study created a hybrid model with LSTM and an adaptive genetic algorithm for different stock indices and showed it's predictive power over other methods (Zeng et al., 2022). 

Overall, the use of a LSTM model is well known for financial forecasting, although it's uses for forecasting real estate models have been limited, especially in the context of Northern Virginia. 


## Description of Data

![Picture of data](https://i.ibb.co/TmZqtWX/Screenshot-2024-05-10-at-7-24-43-PM.png)

My data is from the Northern Virginia Association of Realtors. It includes monthly data from January 2000 to March of 2024, giving 291 total observations. The data covers Arlington County, Alexandria County, Fairfax City, Fairfax County, Loudoun County, and Prince William County. 

Notably, Loudoun County and Prince William County are excluded from the series average of for Northern Virginia. Both counties have seen rampant rises in housing prices as well as significant growth in the past 20 years, however, they were less economically integrated when the series began and was excluded for this reason. These counties data are included separately, and could be incorporated if desired. 

While the geographical region of Northern Virginia would also include counties as far west as Winchester County and as far South as Spotsylvania County, a significant contributor to the expensive home prices in the data series is due to the proximity to D.C. , major defense contracting firms, and Fortune 500 companies headquartered in Arlington, Fairfax, and Alexandria. For that reason, the series distinguishes the more suburban, expensive counties within the broader geographical area. 
   

Variables included:
-   Units sold
-   Average List Price
-   Average Sold Price
-   Ratio of List Price to Sold Price
-   Days on the Market

Each of these five variables was further broken down by number of bedrooms (two bedrooms or less, 3 bedrooms, or four or more bedrooms), if the housing unit was attached or detached, and if the unit was a condo.

## Methods
### Pre-processing methods
Housing data typically exhibits seasonality and this was true for all of the variables in my data set. Below, two of the eight variables are plotted before adjusting for seasonality to illustrate this. To address this, I employed the Seasonal-Trend decomposition using Loess (STL) module from statsmodels.tsa.seasonal. Since my data is monthly, I specified 12 periods to capture the seasonality. Additionally, I used a MinMaxScalar for the data. Finally, I converted the series to supervised learning and split the data into a test and a train portion. 

![Nova Avg Housing Price](https://i.ibb.co/W6QY2Xt/Image-5-10-24-at-11-34-PM.jpg)

![Nova New Listings](https://i.ibb.co/mSdhkgj/Image-5-10-24-at-11-35-PM.jpg)


### The machine learning methods

My primary method employed a Long-Short Term Memory (LSTM) neural network to tune for multiple regression weights. After adjusting for seasonality, I had 18 dependent variables, so my input layer had 18 nodes. The second layer had 14 nodes and I utilized the Rectified Linear Unit (ReLU). I then had a hidden layer with 9 nodes and an output layer with 18 nodes. Below is a simplified version of the neural network architecture I used. 

![](https://mermaid.ink/img/pako:eNrtlcFqAjEQhl9lmLMehB7aPRTULbQgFNp62ngIm7Ebyo5LOqGI-O7GZEWEltb1VNkMhCTz_QPJDJkNlitDmKHivb073VTwliuGMMbFEzdeYKbX5ODLSgWjW-Ag-FzAcHgPk-LRGkN8Qty0RIoxieC0eKFS7NKSgZll0g7mbGURgIT9wrXQWeDZcCdBZ9FFwovFfwmQLBZCynabq3HcXIfneMc0T6Mn_6au707Kuue6c4cXzyP7UDx7-emX6cEe_Mfg4X9RjAOsydXamtBrN3uvQqmoJoVZWBrtPlTowdvAaS-r1zWXmInzNEDfGC2UWx2ac50OtzuZsjiD?type=png)

Additionally, I performed a multiple linear regression as a benchmark to evaluate the LSTM model. 

### Application and validation procedure
I had 291 monthly observations. To train the data, I used 14 years of data, which is roughly 60% of my observations. The 2007-08 housing crash was the only period where housing prices declined, so I gave allocated several years of data after this event to minimize its impacts on the final forecast. The remaining 40% of observations were used to compare the predicted vs actual outcome. 


## Discussion

The LSTM model vastly outperformed the linear regression. Even a relatively simple LSTM neural network had a very low mean squared error, was superior to the regression model, and was not computationally difficult. After tuning for 150 epochs, the model achieved an mse of 0.0329 on the test data. Moreover, the LSTM  model preformed well against the test data set and had relatively low mean squared errors. For the multiple linear regression, however, I am suspicious of overfitting.  The $R^2$ or the regression was 1.00, which is a clear statistical red flag. Even with accounting for a suspicious amount of variation in the data, this model preformed worse than the LSTM model. 

Interestingly, my LSTM model did have one major concern. During the COVID-19 Pandemic, when listings and new construction dropped and days on the market increased, my LSTM model predicted a sharp decline in the average sales price. In reality,  there was hardly ever a drop in the average housing price in the series I examined. The regression model also suffered from this, so it is possible that it is not a default of the method, but a function of the data. 

This concept could be further developed by adding interest rates as increases can disincentive selling and make buying more expensive. Additionally, one of my models major defaults was the exclusion of Loudoun and Prince William counties. This would be easily remedied by adding them to the model, as the data is already provided by NVAR. Overall, there is not an abundance of LSTM models in forecasting real estate, despite LSTM's applicability to many financial settings. It is a useful tool that should be incorporated within the machine learning toolbox. 

## References
1. Borovykh, A., Bohte, S., & Oosterlee, C. W. (2018). Dilated convolutional neural networks for time series forecasting. Journal of Computational Finance. https://doi.org/10.21314/jcf.2019.358
2. Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. European Journal of Operational Research, 270(2), 654–669. [https://doi.org/10.1016/j.ejor.2017.11.054](https://doi.org/10.1016/j.ejor.2017.11.054)
3. Northern Virginia Association of Realtors. Historical Monthly Data. www.nvar.com. https://www.nvar.com/realtors/news/market-statistics/historical-monthly-data
4. Hu, L., Chun, Y., & Griffith, D. A. (2022). Incorporating spatial autocorrelation into house sale price prediction using Random Forest Model. Transactions in GIS, 26(5), 2123–2144. [https://doi.org/10.1111/tgis.12931](https://doi.org/10.1111/tgis.12931)
5. Long & Foster. Northern Virginia Housing Market Data. Long & Foster - Real Estate Market Minute. https://marketminute.longandfoster.com/market-minute/va/northern-virginia.html
6. Louati, A., Lahyani, R., Aldaej, A., Aldumaykhi, A., & Otai, S. (2021). Price forecasting for real estate using Machine Learning: A case study on Riyadh City. Concurrency and Computation: Practice and Experience, 34(6). https://doi.org/10.1002/cpe.6748
7. Mohamed, H. H., Ibrahim, A. H., & A. Hagras, O. (2023). Forecasting the real estate housing prices using a novel Deep Learning Machine Model. Civil Engineering Journal, 9, 46–64. https://doi.org/10.28991/cej-sp2023-09-04
8. Pai, P.-F., & Wang, W.-C. (2020). Using machine learning models and actual transaction data for predicting real estate prices. Applied Sciences, 10(17), 5832. https://doi.org/10.3390/app10175832
9. Wang, X., Wen, J., Zhang, Y., & Wang, Y. (2014). Real estate price forecasting based on SVM optimized by PSO. Optik, 125(3), 1439–1443. https://doi.org/10.1016/j.ijleo.2013.09.017
10. Zeng, X., Cai, J., Liang, C., & Yuan, C. (2022). A hybrid model integrating long short-term memory with adaptive genetic algorithm based on individual ranking for stock index prediction. PLOS ONE, 17(8). https://doi.org/10.1371/journal.pone.0272637



> Written with [StackEdit](https://stackedit.io/).

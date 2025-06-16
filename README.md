# evaluating-moment-open-source-forecasting-applied-to-stock-prices
# Evaluating MOMENT: An Open Source Foundation Model for General- Purpose Time Series Forecasting Applied to Stock Prices
The revolution brought by artificial intelligence has changed the way each person sees the world. From businesspeople who view AI as a way to improve both production and administrative processes, to the academic world where it has made a huge impact—expectations are that, in the coming years, it will be possible to significantly increase the number of publications using data that was previously considered inaccessible. And of course, daily life has changed as well: from the way a student does homework to how anyone can now access any recipe in the world from the palm of their hand.

Focusing on the field of artificial intelligence development, new breakthroughs are reported every week—ranging from leading tech companies like Google to universities that are beginning to acquire the computational capabilities needed to run such models. Currently, large language models (LLMs) are at the forefront of public and academic discourse. Previously, AI-generated images captured the spotlight, and before that, synthetic audio made headlines. There is an ample evidence demonstrating what AI can achieve when it has access to vast datasets and computational power required to bring such projects to fruition.

While artificial intelligence has demonstrated remarkable progress in domains such as image, text, and audio generation, one area that has comparatively lagged behind is time series analysis. There are numerous tools available for working with time series data. Traditionally, one of the most common approaches has involved using statistical models such as ARIMA, which forecast future values based on linear trends. With the rise of deep learning, more complex alternatives have emerged—particularly recurrent neural networks like LSTM, which are capable of capturing intricate temporal dependencies.

However, a compelling question arises: what if we applied architectures similar to those powering today’s most advanced AI systems—namely, transformers—but adapted specifically for time series data rather than conventional tasks such as text processing?

Fortunately, a concrete step in this direction already exists, under the name MOMENT. Developed in 2024 by Auton Lab at Carnegie Mellon University, the model was created by researchers Monomito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, and Artur Dubrawski. MOMENT introduces the first family of open foundation models specifically designed for time series analysis. Its key innovation lies in its ability to perform four core tasks on any time series dataset: forecasting, classification, anomaly detection, and missing data imputation.

Before discussing the purpose of our small experiment, it is essential to gain a deeper understanding of what MOMENT actually is. While we know that it is a tool designed for time series analysis, several key questions remain: How does it work? How was it developed? And how accurate is it compared to existing approaches?

Large pre-trained models in language, vision, and video domains typically perform well across a wide range of tasks and data sources, often with little or no supervision. Moreover, they can be fine-tuned to excel at specific tasks. MOMENT brings these core capabilities to the domain of time series data. This means it can serve as a foundational building block for a wide array of time series analysis tasks. Specifically, MOMENT models:

- Perform effectively out of the box, with minimal or no task-specific examples (enabling capabilities such as zero-shot forecasting and few-shot classification), and
- Can be fine-tuned using in-distribution and task-specific data to further enhance their performance.

MOMENT is a family of high-capacity transformer models, pre-trained using a masked time series prediction task on large amounts of time series data drawn from diverse domains (Goswami et al., 2024).

The dataset known as Time Series Pile encapsulates a diverse collection of time series data. It includes 9 benchmark datasets designed for evaluating long-horizon forecasting, as well as 100,000-time series for short-horizon forecasting. In addition, it comprises over 2,000-time series spanning a wide range of domains, including healthcare, environmental data, and web server logs.

MOMENT is trained on this Time Series Pile, and according to the authors' paper, it achieves significantly better results than both traditional models like ARIMA and more recent deep learning approaches such as TimesNet, which share the same goal of time series analysis.

## Purpose

A review of existing applications of MOMENT in forecasting tasks shows that a significant number is focused on the healthcare domain, such as electrocardiogram (ECG) signal prediction. On the other hand, there is also strong adoption in industrial settings. A notable example provided by the authors on their official GitHub page involves forecasting electricity transformer loads, illustrating the model’s practical relevance across sectors.

Having reviewed the impressive performance of MOMENT, and recognizing that the future of time series analysis lies in understanding the functioning of such advanced algorithms, we propose a small-scale experiment to evaluate MOMENT's ability to predict the stock price of a specific company. This is a particularly challenging task due to the high volatility and complexity inherent in financial time series, which often demand more sophisticated modeling approaches

In other words, rather than feeding MOMENT with complex equations and multiple explanatory variables—which would likely improve its performance—we aim to take a simplified approach, using only the historical stock price data to assess its forecasting capabilities. This approach seeks to evaluate how well MOMENT performs by leveraging its extensive pretraining, even with minimal input information.

For this experiment, we employed a dataset downloaded from Kaggle (<https://www.kaggle.com/datasets/ramamet4/nse-stocks-database>) titled "NSE India stocks (Indices)". This dataset represents the benchmark stock market index of the National Stock Exchange of India and comprises a well-diversified index of 50 stocks spanning 22 sectors of the economy. It is widely used for various purposes, including benchmarking fund portfolios, index-based derivatives, and index funds.

This dataset was selected primarily due to its extensive volume of data, which makes it particularly suitable for our experimental setup. However, it should be noted that this methodology is not limited to the NSE index; one could equally apply it to other stock indices, such as those of Apple or Microsoft, to assess forecasting capabilities.

## Methodology

To implement the algorithm, several design decisions must be made depending on the nature of the data and the specific forecasting objectives. One such constraint defined by the model and its authors is the fixed input length of 512 time steps for training. However, an important parameter left open to the user is the forecasting horizon. In the original paper, a figure on the first page highlights MOMENT’s performance across different task categories, particularly in short-term forecasting and long-horizon forecasting, having better results in the last one.

In line with this, we will evaluate MOMENT’s forecasting ability across multiple time horizons:

- 60-time horizon
- 120-time horizon
- 180-time horizon
- 240- time horizon

Before training the model with different forecasting horizons, it is important to understand the model architecture. In summary, MOMENT divided time series into disjoint fixed-length sub-sequences called patches, and each patch is mapped into a D-dimensional patch embedding. During pretraining, patches are randomly masked by replacing their embeddings with a special mask token, \[MASK\]. The goal of pretraining is to learn patch embeddings that can be used to reconstruct the original time series using a lightweight reconstruction head. (Goswami et al., 2024).

Furthermore, it is important to highlight that there are two primary fine-tuning strategies that can be applied to MOMENT, regardless of the prediction window:

1. Head-only fine-tuning, where only the final layers (the “head”) of the model are updated using the new data, while all other parameters remain frozen. This approach assumes that the pretrained backbone has already learned general time series patterns and only requires minimal adaptation to the specific dataset.
2. Full fine-tuning, where all model parameters are unfrozen and allowed to adapt to the new data. While this strategy can potentially lead to better performance, it is also more computationally intensive and may risk overfitting if not carefully managed.

In summary, we will conduct a comparative analysis using MSE and MAE as evaluation metrics, complemented by visualizations, to highlight the differences between the two fine-tuning strategies (head-only and full fine-tuning) across various forecasting horizons.

The goal is to assess the effectiveness of the MOMENT model in predicting stock prices, a notoriously challenging and volatile domain within time series forecasting.

## Results

After conducting 75 epochs for each forecasting horizon and fine-tuning method, we obtained the following results:
| Forecast Horizon | Head-only Fine Tuning | Full Fine Tuning |
|------------------|-----------------------|------------------|
|                  | MSE       | MAE       | MSE       | MAE       |
| 60               | .021      | .089      | .028      | .106      |
| 120              | .039      | .124      | .048      | .150      |
| 180              | .056      | .151      | .068      | .170      |
| 240              | .072      | .174      | .095      | .195      |

Table Resume of results made by the algorithm. Source: Author's work

The visualizations can be accessed by reviewing the code where the models are applied. In addition to viewing the graphs, this code can also serve as a foundation for further experiments or custom calculations.

## Conclusion

Upon reviewing both the numerical results and the visualizations, it becomes evident that the model still struggles to learn consistently. However, its performance improves significantly when the forecast horizon is reduced. For instance, in a test using only one prediction time step, the model achieved an MSE of 0.002 and an MAE of 0.031—substantially outperforming the results shown in Table 1.

This also suggests that full fine-tuning may not be necessary, as better results were achieved by tuning only the head—highlighting the strength of the MOMENT architecture. This demonstrates the model's ability to understand time series patterns effectively. Finally, due to the inherent difficulty of anticipating stock price fluctuations, a forecast horizon of 60 days or fewer appears to be optimal—at least in this specific case. It is also worth noting that this analysis relied solely on historical stock prices, without incorporating any exogenous variables that could potentially enhance prediction accuracy. Therefore, a logical next step would be to incorporate such variables to further improve MOMENT's performance—something the model is designed to support.

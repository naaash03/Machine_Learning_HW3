import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

file_path = "Auto.csv"
auto_data = pd.read_csv(file_path)

auto_data["horsepower"] = pd.to_numeric(auto_data["horsepower"], errors="coerce")
auto_data = auto_data.dropna()

auto_data.head()

x = auto_data["horsepower"]
y = auto_data["mpg"]

x = sm.add_constant(x)

model = sm.OLS(y,x).fit()

print(model.summary())

new_hp = pd.DataFrame({'const': 1, 'horsepower': [98]})         #defining new data points where horsepower =  98

predicted_mpg = model.predict(new_hp)[0]                        #predicting the new mpg

conf_int = model.get_prediction(new_hp).conf_int()              #obtain the confidence intervals

pred_int = model.get_prediction(new_hp).summary_frame(alpha=0.05)[["mean_ci_lower","mean_ci_upper","obs_ci_lower","obs_ci_upper"]]          #Obtain prediction intervals

print(f"Predicted mpg for horsepower = 98: {predicted_mpg:.2f}")
print("95% Confidence Interval:", conf_int)
print("95% Prediction Interval:", pred_int)


#plotting the line of regression
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(x=auto_data["horsepower"], y=auto_data["mpg"], ax=ax)

x_range = np.linspace(auto_data["horsepower"].min(), auto_data["horsepower"].max(), 100)
y_pred = model.params["const"] + model.params["horsepower"] * x_range
ax.plot(x_range, y_pred, color="red", linewidth=2, label="Regression Line")

ax.set_xlabel("Horsepower")
ax.set_ylabel("MPG")
ax.set_title("Regression of MPG on Horsepower")
ax.legend()
plt.show


##Diagnostic Plot

fig, ax = plt.subplots(figsize=(8,6))
sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, ax=ax, line_kws={"color": "red"})
ax.axhline(0, color='black', linestyle='--')
ax.set_xlabel("Fitted Values")
ax.set_ylabel("Residuals")
ax.set_title("Residual Plot")
plt.show

##normality check for the residuals
sm.qqplot(model.resid, line='45', fit = True)
plt.title("QQ Plot of Residuals")
plt.show()
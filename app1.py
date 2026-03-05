from flask import Flask, render_template, request
import numpy as np
from scipy.stats import t
from statistics import stdev

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():

    result = None

    if request.method == "POST":
        try:
            sample1 = request.form.get("sample1")
            sample2 = request.form.get("sample2")
            alternative = request.form.get("alternative")

            # Convert input into lists
            a = list(map(float, sample1.split(",")))
            b = list(map(float, sample2.split(",")))

            n1 = len(a)
            n2 = len(b)

            xbar1 = np.mean(a)
            xbar2 = np.mean(b)

            sd1 = stdev(a)
            sd2 = stdev(b)

            df = n1 + n2 - 2

            se = np.sqrt((sd1**2 / n1) + (sd2**2 / n2))

            tcal = (xbar1 - xbar2) / se

            # p-value calculation
            if alternative == "two":
                p_value = 2 * (1 - t.cdf(abs(tcal), df))
            elif alternative == "left":
                p_value = t.cdf(tcal, df)
            else:
                p_value = 1 - t.cdf(tcal, df)

            decision = "Reject the Null Hypothesis" if p_value < 0.05 else "Fail to Reject the Null Hypothesis"

            result = {
                "mean1": round(xbar1, 4),
                "mean2": round(xbar2, 4),
                "t": round(tcal, 4),
                "df": df,
                "p": round(p_value, 6),
                "decision": decision
            }

        except:
            result = {"error": "Please enter valid numbers separated by commas."}

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
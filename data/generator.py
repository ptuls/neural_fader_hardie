import pandas as pd
import numpy as np


def main():
    # Parameters
    n_rows = 50_000
    n_cohorts = 5

    # Cohort month generation
    users_per_cohort = n_rows // n_cohorts
    cohort_months = np.array([[i] * users_per_cohort for i in range(1, n_cohorts + 1)]).flatten()

    # Country distribution
    countries = ["US", "GB", "CA", "AU", "DE"]
    country_probs = [0.45, 0.20, 0.15, 0.15, 0.05]
    country_column = np.random.choice(countries, n_rows, p=country_probs)

    # Plan names
    plans = ["Basic", "Pro", "Premium"]
    plan_column = np.random.choice(plans, n_rows)

    # Country distribution
    payment_gateway = ["Stripe", "Adyen"]
    payment_gateway_probs = [0.6, 0.4]
    payment_gateway_column = np.random.choice(payment_gateway, n_rows, p=payment_gateway_probs)

    # Churn rate generation based on plan
    initial_churn_rate = 0.05
    churn_rate_increase = 0.05
    churn_rate_basic = [
        initial_churn_rate + 0.07 + churn_rate_increase * (i - 1) for i in range(1, n_cohorts + 1)
    ]
    churn_rate_pro = [
        initial_churn_rate + 0.04 + churn_rate_increase * (i - 1) for i in range(1, n_cohorts + 1)
    ]
    churn_rate_premium = [
        initial_churn_rate + churn_rate_increase * (i - 1) for i in range(1, n_cohorts + 1)
    ]

    # Assign churn based on plan and cohort
    churn_column = np.zeros(n_rows)
    for i in range(1, n_cohorts + 1):
        churn_column[(cohort_months == i) & (plan_column == "Basic")] = np.random.choice(
            [0, 1],
            p=[1 - churn_rate_basic[i - 1], churn_rate_basic[i - 1]],
            size=sum((cohort_months == i) & (plan_column == "Basic")),
        )
        churn_column[(cohort_months == i) & (plan_column == "Pro")] = np.random.choice(
            [0, 1],
            p=[1 - churn_rate_pro[i - 1], churn_rate_pro[i - 1]],
            size=sum((cohort_months == i) & (plan_column == "Pro")),
        )
        churn_column[(cohort_months == i) & (plan_column == "Premium")] = np.random.choice(
            [0, 1],
            p=[1 - churn_rate_premium[i - 1], churn_rate_premium[i - 1]],
            size=sum((cohort_months == i) & (plan_column == "Premium")),
        )

    # Numerical features: num_product_uses, avg_session_time, num_searches, age
    num_product_uses = np.random.randint(1, 101, n_rows)
    num_product_uses[churn_column == 1] = num_product_uses[churn_column == 1] * np.random.uniform(
        0.5, 0.8, sum(churn_column == 1)
    ).astype(int)

    avg_session_time = np.random.randint(5, 121, n_rows)
    avg_session_time[churn_column == 1] = avg_session_time[churn_column == 1] * np.random.uniform(
        0.5, 0.8, sum(churn_column == 1)
    ).astype(int)

    num_searches = np.random.randint(0, 51, n_rows)
    num_searches[churn_column == 1] = num_searches[churn_column == 1] * np.random.uniform(
        0.3, 0.6, sum(churn_column == 1)
    ).astype(int)

    age = np.random.randint(18, 71, n_rows)

    # Construct the dataframe
    df = pd.DataFrame(
        {
            "user_id": range(1, n_rows + 1),
            "cohort_month": cohort_months,
            "country": country_column,
            "payment_gateway": payment_gateway_column,
            "plan_name": plan_column,
            "age": age,
            "num_product_uses": num_product_uses,
            "num_searches": num_searches,
            "avg_session_time": avg_session_time,
            "churned": churn_column,
        }
    )
    df.to_csv("data/dataset.csv", index=False)


if __name__ == "__main__":
    main()

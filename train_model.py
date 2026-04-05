{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f459ea8d-22ce-492f-94c9-9eccbee53405",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model saved as churn_model.pkl\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "import joblib\n",
    "\n",
    "# --- Sample dataset (you can swap in your own) ---\n",
    "data = pd.DataFrame({\n",
    "    \"monthly_spend\":  [50, 60, 70, 120, 150, 200, 80, 90, 40, 30],\n",
    "    \"tenure_months\":  [ 2,  3,  6,  12,  24,  36,  4,  5,  1,  1],\n",
    "    \"support_calls\":  [ 3,  1,  0,   1,   0,   0,  2,  1,  4,  3],\n",
    "    \"churn\":          [ 1,  1,  0,   0,   0,   0,  1,  0,  1,  1]\n",
    "})\n",
    "\n",
    "# --- Prepare features and target ---\n",
    "X = data[[\"monthly_spend\", \"tenure_months\", \"support_calls\"]]\n",
    "y = data[\"churn\"]\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n",
    "\n",
    "# --- Train the model ---\n",
    "model = DecisionTreeClassifier()\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# IN PRACTICE (or final projects), create the best performing model you can (balancing bias-variance) using all your skills. \n",
    "\n",
    "# --- NEW STEP: Save the trained model to a file ---\n",
    "# --- This is the DEPLOYED model ---\n",
    "joblib.dump(model, \"churn_model.pkl\")\n",
    "\n",
    "print(\"Model saved as churn_model.pkl\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "51ae0a3e-300d-44d7-a51b-b404e855585b",
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'fastapi'",
     "output_type": "error",
     "traceback": [
      "\u001b[31m---------------------------------------------------------------------------\u001b[39m",
      "\u001b[31mModuleNotFoundError\u001b[39m                       Traceback (most recent call last)",
      "\u001b[36mCell\u001b[39m\u001b[36m \u001b[39m\u001b[32mIn[1]\u001b[39m\u001b[32m, line 1\u001b[39m\n\u001b[32m----> \u001b[39m\u001b[32m1\u001b[39m \u001b[38;5;28;01mfrom\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mfastapi\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mimport\u001b[39;00m FastAPI\n\u001b[32m      2\u001b[39m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mjoblib\u001b[39;00m\n\u001b[32m      3\u001b[39m \u001b[38;5;28;01mimport\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mnumpy\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mas\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[34;01mnp\u001b[39;00m\n",
      "\u001b[31mModuleNotFoundError\u001b[39m: No module named 'fastapi'"
     ]
    }
   ],
   "source": [
    "from fastapi import FastAPI\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "# --- Create the API application ---\n",
    "app = FastAPI(title=\"Churn Prediction API\")\n",
    "\n",
    "# --- Load the saved model (from the .pkl file) ---\n",
    "model = joblib.load(\"churn_model.pkl\")\n",
    "\n",
    "# --- Endpoint 1: Home page (just confirms the API is alive) ---\n",
    "@app.get(\"/\")\n",
    "def home():\n",
    "    return {\"message\": \"Churn Prediction API is running\"}\n",
    "\n",
    "# --- Endpoint 2: Prediction (this is the one that does the work) ---\n",
    "@app.post(\"/predict\")\n",
    "def predict(data: dict):\n",
    "    # Pull the customer's features out of the incoming request\n",
    "    features = np.array([[\n",
    "        data[\"monthly_spend\"],\n",
    "        data[\"tenure_months\"],\n",
    "        data[\"support_calls\"]\n",
    "    ]])\n",
    "\n",
    "    # Use the model to get a churn probability\n",
    "    churn_prob = model.predict_proba(features)[0][1]\n",
    "\n",
    "    return {\n",
    "        \"churn_probability\": round(float(churn_prob), 4)\n",
    "    }"
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

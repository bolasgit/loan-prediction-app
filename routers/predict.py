import sys
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

sys.path.append("..")

from starlette import status
from starlette.responses import RedirectResponse
from fastapi import Depends, APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

LR = LogisticRegression()

router = APIRouter(
    prefix="/predict",
    tags=["Loan Predictor"],
    responses={404: {"description": "Not found"}},
)


templates = Jinja2Templates(directory="templates")


@router.get("/", response_class=HTMLResponse)
async def view_predict_form(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})


@router.post("/predict", response_class=HTMLResponse)
async def predict_loan_status(
    request: Request,
    gender=Form(...),
    married=Form(...),
    dependents=Form(...),
    education=Form(...),
    loan_term=Form(...),
    credit_history=Form(...),
    loan_amount=Form(...),
    personal_income=Form(...),
    spouse_income=Form(...),
):
    gender = 0 if gender != "male" else 1
    married = 0 if married != "Yes" else 1
    credit_history = 0 if credit_history != "Yes" else 1
    if dependents in ("1", "2"):
        dependents = int(dependents) - 1
    else:
        dependents = 2
    education = 0 if education != "Yes" else 1

    input_params = []
    gender = int(gender)
    married = int(married)
    dependents = int(dependents)
    education = int(education)
    loan_term = int(loan_term)
    credit_history = int(credit_history)
    loan_amount_log = np.log(int(loan_amount))
    total_income = int(personal_income) + int(spouse_income)

    file = "models\ML_Model2.pkl"
    with open(file, "rb") as f:
        k = pickle.load(f)

    input_params = []
    input_params.extend(
        [
            gender,
            married,
            dependents,
            education,
            loan_term,
            credit_history,
            loan_amount_log,
            total_income,
        ]
    )

    # Combine user input into a feature vector
    input_data = np.array([input_params])

    # Make prediction using the trained model
    loan_request_status = k.predict(input_data)[0]

    if loan_request_status == 1:
        msg = "Congratulations your loan has been approved!"
    else:
        msg = "I am sorry to inform you that your loan request has been declined"

    return templates.TemplateResponse(
        "prediction.html", {"request": request, "msg": msg}
    )


# def predict_loan_status(params):
#     # params = [Gender, Married, Dependents, Education, Loan_term, Credit_history, loan_amount_log, total_income]
#     # Convert user input to log
#     file = "\models\ML_Model2.pkl"
#     with open(file, "rb") as f:
#         k = pickle.load(f)

#     input_params = []
#     gender = params[0]
#     married = params[1]
#     dependents = params[2]
#     education = params[3]
#     loan_term = params[4]
#     credit_history = params[5]
#     loan_amount_log = np.log(params[6])
#     total_income = params[7] + params[8]

#     input_params.extend(
#         [
#             gender,
#             married,
#             dependents,
#             education,
#             loan_term,
#             credit_history,
#             loan_amount_log,
#             total_income,
#         ]
#     )

#     # Combine user input into a feature vector
#     input_data = np.array([input_params])

#     # Make prediction using the trained model
#     loan_request_status = k.predict(input_data)[0]

#     # Return the predicted price
#     return loan_request_status

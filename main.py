from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pymysql


db_config = {
    'host': 'localhost', 
    'user': 'root',      
    'password': 'root', 
    'database': 'statlog',
}

connection = pymysql.connect(**db_config)

try:
    print("Conexão bem-sucedida!")
except Exception as e:
    print(f"Erro ao conectar: {e}")
finally:
    connection.close()

connection = pymysql.connect(**db_config)
query = "SELECT * FROM germancredit;"
df = pd.read_sql(query, connection)
connection.close()


if 'id' in df.columns:
    df = df.drop('id', axis=1)


print(df.head())
print(df.info())
print(df.describe())

mappings = {
    'laufkont': {1: 'no checking account', 2: '... < 0 DM', 3: '0<= ... < 200 DM', 4: '... >= 200 DM / salary for at least 1 year'},
    'moral': {0: 'delay in paying off in the past', 1: 'critical account/other credits elsewhere', 2: 'no credits taken/all credits paid back duly', 3: 'existing credits paid back duly till now', 4: 'all credits at this bank paid back duly'},
    'verw': {0: 'others', 1: 'car (new)', 2: 'car (used)', 3: 'furniture/equipment', 4: 'radio/television', 5: 'domestic appliances', 6: 'repairs', 7: 'education', 8: 'vacation', 9: 'retraining', 10: 'business'},
    'sparkont': {1: 'unknown/no savings account', 2: '... < 100 DM', 3: '100 <= ... < 500 DM', 4: '500 <= ... < 1000 DM', 5: '... >= 1000 DM'},
    'beszeit': {1: 'unemployed', 2: '< 1 yr', 3: '1 <= ... < 4 yrs', 4: '4 <= ... < 7 yrs', 5: '>= 7 yrs'},
    'rate': {1: '>= 35', 2: '25 <= ... < 35', 3: '20 <= ... < 25', 4: '< 20'},
    'famges': {1: 'male : divorced/separated', 2: 'female : non-single or male : single', 3: 'male : married/widowed', 4: 'female : single'},
    'buerge': {1: 'none', 2: 'co-applicant', 3: 'guarantor'},
    'wohnzeit': {1: '< 1 yr', 2: '1 <= ... < 4 yrs', 3: '4 <= ... < 7 yrs', 4: '>= 7 yrs'},
    'verm': {1: 'unknown / no property', 2: 'car or other', 3: 'building soc. savings agr./life insurance', 4: 'real estate'},
    'weitkred': {1: 'bank', 2: 'stores', 3: 'none'},
    'wohn': {1: 'for free', 2: 'rent', 3: 'own'},
    'bishkred': {1: '1', 2: '2-3', 3: '4-5', 4: '>= 6'},
    'beruf': {1: 'unemployed/unskilled - non-resident', 2: 'unskilled - resident', 3: 'skilled employee/official', 4: 'manager/self-empl./highly qualif. employee'},
    'pers': {1: '3 or more', 2: '0 to 2'},
    'telef': {1: 'no', 2: 'yes (under customer name)'},
    'gastarb': {1: 'yes', 2: 'no'},
    'kredit': {0: 'bad', 1: 'good'}
}

column_mapping = {
    'laufkont': 'status',
    'laufzeit': 'duration',
    'moral': 'credit_history',
    'verw': 'purpose',
    'hoehe': 'amount',
    'sparkont': 'savings',
    'beszeit': 'employment_duration',
    'rate': 'installment_rate',
    'famges': 'personal_status_sex',
    'buerge': 'other_debtors',
    'wohnzeit': 'present_residence',
    'verm': 'property',
    'alter': 'age',
    'weitkred': 'other_installment_plans',
    'wohn': 'housing',
    'bishkred': 'number_credits',
    'beruf': 'job',
    'pers': 'people_liable',
    'telef': 'telephone',
    'gastarb': 'foreign_worker',
    'kredit': 'credit_risk'
}

df.rename(columns=column_mapping, inplace=True)

for column, mapping in mappings.items():
    if column in column_mapping:
        df[column_mapping[column]] = df[column_mapping[column]].map(mapping)

print(df.head())
df.to_csv('germancredit_translated.csv', index=False, sep=';')
print("Dados traduzidos salvos em 'germancredit_translated.csv'")


print("Balanceamento das classes:")
print(df['credit_risk'].value_counts(normalize=True))


X = df.drop('credit_risk', axis=1)
y = df['credit_risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)


model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

import tkinter as tk
from tkinter import ttk, messagebox

def tk_predict():
    def fazer_previsao():
        try:
            duration_val = int(duration_entry.get())
            amount_val = int(amount_entry.get())
            age_val = int(age_entry.get())
        except ValueError:
            messagebox.showerror("Erro", "Preencha os campos numéricos corretamente.")
            return

        dados = {col: [0] for col in X_train.columns}
        if 'duration' in dados: dados['duration'] = [duration_val]
        if 'amount' in dados: dados['amount'] = [amount_val]
        if 'age' in dados: dados['age'] = [age_val]
        dados[credit_history_var.get()] = [1]
        dados[personal_status_sex_var.get()] = [1]
        dados[property_var.get()] = [1]
        dados[job_var.get()] = [1]
        dados[purpose_var.get()] = [1]
        dados[savings_var.get()] = [1]

        dados_df = pd.DataFrame(dados)[X_train.columns]
        resultado = model.predict(dados_df)[0]
        texto = " Mau crédito" if resultado == 'bad' else " Bom crédito"
        result_label.config(text=f"Resultado: {texto}")

    root = tk.Tk()
    root.title("Previsão de Crédito")

    tk.Label(root, text="Duração (4-72):").grid(row=0, column=0)
    duration_entry = tk.Entry(root)
    duration_entry.insert(0, "24")
    duration_entry.grid(row=0, column=1)

    tk.Label(root, text="Valor (250-20000):").grid(row=1, column=0)
    amount_entry = tk.Entry(root)
    amount_entry.insert(0, "1000")
    amount_entry.grid(row=1, column=1)

    tk.Label(root, text="Idade (18-75):").grid(row=2, column=0)
    age_entry = tk.Entry(root)
    age_entry.insert(0, "30")
    age_entry.grid(row=2, column=1)

    def get_opts(prefix):
        return [col for col in X_train.columns if col.startswith(prefix)]


    credit_history_var = tk.StringVar(value=get_opts('credit_history_')[0])
    personal_status_sex_var = tk.StringVar(value=get_opts('personal_status_sex_')[0])
    property_var = tk.StringVar(value=get_opts('property_')[0])
    job_var = tk.StringVar(value=get_opts('job_')[0])
    purpose_var = tk.StringVar(value=get_opts('purpose_')[0])
    savings_var = tk.StringVar(value=get_opts('savings_')[0])

    ttk.Label(root, text="Hist. crédito:").grid(row=3, column=0)
    ttk.Combobox(root, textvariable=credit_history_var, values=get_opts('credit_history_'), width=40).grid(row=3, column=1)

    ttk.Label(root, text="Sexo/Estado:").grid(row=4, column=0)
    ttk.Combobox(root, textvariable=personal_status_sex_var, values=get_opts('personal_status_sex_'), width=40).grid(row=4, column=1)

    ttk.Label(root, text="Propriedade:").grid(row=5, column=0)
    ttk.Combobox(root, textvariable=property_var, values=get_opts('property_'), width=40).grid(row=5, column=1)

    ttk.Label(root, text="Profissão:").grid(row=6, column=0)
    ttk.Combobox(root, textvariable=job_var, values=get_opts('job_'), width=40).grid(row=6, column=1)

    ttk.Label(root, text="Propósito:").grid(row=7, column=0)
    ttk.Combobox(root, textvariable=purpose_var, values=get_opts('purpose_'), width=40).grid(row=7, column=1)

    ttk.Label(root, text="Poupança:").grid(row=8, column=0)
    ttk.Combobox(root, textvariable=savings_var, values=get_opts('savings_'), width=40).grid(row=8, column=1)

    tk.Button(root, text="Fazer previsão", command=fazer_previsao).grid(row=9, column=0, columnspan=2, pady=10)
    result_label = tk.Label(root, text="Resultado: ")
    result_label.grid(row=10, column=0, columnspan=2)

    root.mainloop()

if __name__ == "__main__":
    tk_predict()
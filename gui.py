import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeRegressor
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "D:\\depi ml\\src\\p1\\p1\\housing.csv")
def train_model():
    df = pd.read_csv(DATA_FILE)
    y = df["MEDV"]
    X = df.drop("MEDV", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)
    grid = GridSearchCV(
        DecisionTreeRegressor(random_state=0),
        param_grid={"max_depth": list(range(1, 11))},
        scoring=make_scorer(r2_score),
        cv=cv
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    stats = {
        "depth": best_model.get_params()["max_depth"],
        "r2": r2_score(y_test, best_model.predict(X_test)),
        "median": y.median(),
        "columns": list(X.columns)
    }
    return best_model, stats


class HousingApp(tk.Tk):
    def __init__(self, model, stats):
        super().__init__()
        self.model = model
        self.stats = stats

        self.title("Boston Housing Estimator")
        self.geometry("400x520")
        self.configure(bg="#1e1e1e")
        self.resizable(False, False)

        self.setup_ui()

    def setup_ui(self):
        # Header
        tk.Label(
            self, text="Boston Housing Valuation", 
            font=("Segoe UI", 13, "bold"), fg="#ffffff", bg="#1e1e1e"
        ).pack(pady=(20, 2))

        sub_txt = f"Model Depth: {self.stats['depth']}  |  Test R²: {self.stats['r2']:.2f}"
        tk.Label(
            self, text=sub_txt, 
            font=("Segoe UI", 9), fg="#888888", bg="#1e1e1e"
        ).pack(pady=(0, 15))

        # Inputs Frame
        form = tk.Frame(self, bg="#252526", padx=15, pady=15)
        form.pack(padx=20, pady=5, fill="x")

        # RM
        tk.Label(form, text="Average Rooms (RM):", font=("Segoe UI", 9), fg="#cccccc", bg="#252526").grid(row=0, column=0, sticky="w", pady=8)
        self.rm_entry = tk.Entry(form, width=12, bg="#3c3c3c", fg="#ffffff", insertbackground="white", bd=1, relief="solid")
        self.rm_entry.grid(row=0, column=1, padx=10, pady=8)

        # LSTAT
        tk.Label(form, text="Poverty Level % (LSTAT):", font=("Segoe UI", 9), fg="#cccccc", bg="#252526").grid(row=1, column=0, sticky="w", pady=8)
        self.lstat_entry = tk.Entry(form, width=12, bg="#3c3c3c", fg="#ffffff", insertbackground="white", bd=1, relief="solid")
        self.lstat_entry.grid(row=1, column=1, padx=10, pady=8)

        # PTRATIO
        tk.Label(form, text="Student-Teacher Ratio:", font=("Segoe UI", 9), fg="#cccccc", bg="#252526").grid(row=2, column=0, sticky="w", pady=8)
        self.ptratio_entry = tk.Entry(form, width=12, bg="#3c3c3c", fg="#ffffff", insertbackground="white", bd=1, relief="solid")
        self.ptratio_entry.grid(row=2, column=1, padx=10, pady=8)

        # Quick Client Shortcuts
        client_frame = tk.Frame(self, bg="#1e1e1e")
        client_frame.pack(pady=10)

        tk.Button(client_frame, text="Client 1", command=lambda: self.set_client(5, 17, 15), bg="#333333", fg="#dddddd", bd=0, padx=8, pady=3, cursor="hand2").pack(side="left", padx=3)
        tk.Button(client_frame, text="Client 2", command=lambda: self.set_client(4, 32, 22), bg="#333333", fg="#dddddd", bd=0, padx=8, pady=3, cursor="hand2").pack(side="left", padx=3)
        tk.Button(client_frame, text="Client 3", command=lambda: self.set_client(8, 3, 12), bg="#333333", fg="#dddddd", bd=0, padx=8, pady=3, cursor="hand2").pack(side="left", padx=3)

        # Calculate Button
        tk.Button(
            self, text="Estimate Price", command=self.on_predict, 
            bg="#007acc", fg="#ffffff", font=("Segoe UI", 10, "bold"), 
            bd=0, cursor="hand2", pady=6
        ).pack(padx=20, pady=15, fill="x")

        # Output Box
        out_frame = tk.Frame(self, bg="#252526", padx=15, pady=12)
        out_frame.pack(padx=20, pady=5, fill="x")

        self.res_lbl = tk.Label(out_frame, text="$0.00", font=("Segoe UI", 16, "bold"), fg="#4ec9b0", bg="#252526")
        self.res_lbl.pack()

        self.diff_lbl = tk.Label(out_frame, text="Select or enter features to calculate", font=("Segoe UI", 8), fg="#777777", bg="#252526")
        self.diff_lbl.pack(pady=(3, 0))

    def set_client(self, rm, lstat, ptratio):
        self.rm_entry.delete(0, tk.END)
        self.rm_entry.insert(0, str(rm))
        self.lstat_entry.delete(0, tk.END)
        self.lstat_entry.insert(0, str(lstat))
        self.ptratio_entry.delete(0, tk.END)
        self.ptratio_entry.insert(0, str(ptratio))
        self.on_predict()

    def on_predict(self):
        try:
            rm = float(self.rm_entry.get())
            lstat = float(self.lstat_entry.get())
            ptratio = float(self.ptratio_entry.get())

            df = pd.DataFrame([[rm, lstat, ptratio]], columns=self.stats["columns"])
            price = float(self.model.predict(df)[0])

            diff = price - self.stats["median"]
            status = "above" if diff >= 0 else "below"

            self.res_lbl.config(text=f"${price:,.2f}")
            self.diff_lbl.config(text=f"${abs(diff):,.0f} {status} dataset median (${self.stats['median']:,.0f})")

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values.")


if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print(f"File {DATA_FILE} not found.")
    else:
        model, stats = train_model()
        app = HousingApp(model, stats)
        app.mainloop()
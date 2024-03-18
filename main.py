import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filename):
    try:
        music_df = pd.read_csv(filename)
        return music_df
    except FileNotFoundError:
        print("El archivo especificado no se encontró.")
        return None

def train_models(X_train, y_train):
    models = {"Logistic Regression": LogisticRegression(),
              "KNN": KNeighborsClassifier(),
              "Decision Tree Classifier": DecisionTreeClassifier()}
    results = {}

    for model_name, model in models.items():
        kf = KFold(n_splits=6, random_state=12, shuffle=True)
        cv_results = cross_val_score(model, X_train, y_train, cv=kf)
        results[model_name] = cv_results

    return results

def plot_results(results):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=pd.DataFrame(results))
    plt.xlabel('Modelos')
    plt.ylabel('Puntuación de Validación Cruzada')
    plt.title('Puntuaciones de Validación Cruzada para Diferentes Modelos')
    plt.xticks(rotation=45)
    plt.savefig('Puntuaciones de Validación Cruzada para Diferentes Modelos.png')
    plt.show()

def main():
    music_df = load_data('music_clean.csv')
    if music_df is None:
        return

    print("Datos cargados exitosamente.")
    print("Cabeza del DataFrame:")
    print(music_df.head(5))
    print("Forma del DataFrame:")
    print(music_df.shape)
    print("Columnas:")
    print(music_df.columns)

    # Convertir variables categóricas a variables dummy
    music_dummies = pd.get_dummies(music_df, drop_first=True)

    print("Forma de music_dummies: {}".format(music_dummies.shape))

    X = music_dummies.drop("genre", axis=1)
    y = music_dummies["genre"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    results = train_models(X_train, y_train)
    plot_results(results)

if __name__ == "__main__":
    main()

import shap
import lime
import lime.lime_tabular
import numpy as np
import matplotlib.pyplot as plt

def global_shap_explanation(classifier, X_features, feature_names, save_path=None):
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_features)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_features,
        feature_names=feature_names,
        show=False
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return shap_values

def local_lime_explanation(
    classifier,
    X_train_features,
    instance,
    feature_names,
    num_features=10
):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_features,
        feature_names=feature_names,
        class_names=["Healthy", "Failure risk"],
        mode="classification",
        discretize_continuous=True
    )

    explanation = explainer.explain_instance(
        instance,
        classifier.predict_proba,
        num_features=num_features
    )

    return explanation
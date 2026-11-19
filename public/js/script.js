
const form = document.getElementById("predictionForm");
const predictButton = document.getElementById("predictButton");

const result = document.getElementById("result");
const error = document.getElementById("error");

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    // Hide previous results and errors
    result.classList.add("hidden");
    error.classList.add("hidden");

    // Collect all 13 patient measurements
    const fields = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal"
    ];

    const patientData = {};

    fields.forEach((field) => {
        patientData[field] = Number(document.getElementById(field).value);
    });

    predictButton.disabled = true;
    predictButton.textContent = "Processing...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(patientData)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed.");
        }

        // Display prediction
        document.getElementById("predictionText").textContent = data.result;

        document.getElementById("confidence").textContent =
            `Model confidence: ${data.confidence}%`;

        document.getElementById("noDisease").textContent =
            `${data.probabilities.no_disease}%`;

        document.getElementById("heartDisease").textContent =
            `${data.probabilities.heart_disease}%`;

        result.classList.remove("hidden");

    } catch (err) {
        error.textContent = err.message;
        error.classList.remove("hidden");

    } finally {
        predictButton.disabled = false;
        predictButton.textContent = "Generate prediction";
    }
});

const resetButton = document.getElementById("resetButton");

resetButton.addEventListener("click", () => {
    form.reset();
    result.classList.add("hidden");
    error.classList.add("hidden");

    document.getElementById("age").focus();
});
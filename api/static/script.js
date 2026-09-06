const form = document.querySelector("#prediction-form");
const button = document.querySelector("#submit-button");
const statusBox = document.querySelector("#status");
const resultBox = document.querySelector("#result");
const priceBox = document.querySelector("#predicted-price");

function numberOrNull(value) {
    if (value === "" || value === null || value === undefined) {
        return null;
    }

    const number = Number(value);
    return Number.isNaN(number) ? null : number;
}

function makePayload(formData) {
    return {
        manufacturer: formData.get("manufacturer"),
        model: formData.get("model") || null,
        year: numberOrNull(formData.get("year")),
        mileage: numberOrNull(formData.get("mileage")),
        engine: formData.get("engine") || null,
        transmission: formData.get("transmission") || null,
        drivetrain: formData.get("drivetrain") || null,
        fuel_type: formData.get("fuel_type") || null,
        mpg: formData.get("mpg") || null,
        exterior_color: formData.get("exterior_color") || null,
        interior_color: formData.get("interior_color") || null,
        accidents_or_damage: numberOrNull(formData.get("accidents_or_damage")),
        one_owner: numberOrNull(formData.get("one_owner")),
        personal_use_only: numberOrNull(formData.get("personal_use_only")),
        driver_rating: numberOrNull(formData.get("driver_rating")),
        driver_reviews_num: numberOrNull(formData.get("driver_reviews_num")),
    };
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    button.disabled = true;
    button.textContent = "Рассчитываем...";
    statusBox.textContent = "";
    resultBox.classList.add("hidden");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(makePayload(new FormData(form))),
        });

        const body = await response.json();

        if (!response.ok) {
            const detail = body.detail || "Сервер вернул ошибку";
            throw new Error(detail);
        }

        const price = Number(body.predicted_price);
        priceBox.textContent = `${price.toLocaleString("ru-RU", {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
        })} ${body.currency || "USD"}`;
        resultBox.classList.remove("hidden");
    } catch (error) {
        statusBox.textContent = `Ошибка: ${error.message}`;
    } finally {
        button.disabled = false;
        button.textContent = "Рассчитать цену";
    }
});

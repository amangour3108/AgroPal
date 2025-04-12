import sys
import joblib
import numpy as np

if len(sys.argv) != 12:
    print(f"Invalid input\nReceived {len(sys.argv) - 1} arguments: {sys.argv[1:]}")
    sys.exit(1)

try:
    mean_temp = float(sys.argv[1])
    max_temp = float(sys.argv[2])
    min_temp = float(sys.argv[3])
    humidity = float(sys.argv[4])
    wind_speed = float(sys.argv[5])
    pressure = float(sys.argv[6])
    radiation = float(sys.argv[7])
    day_of_year = int(sys.argv[8])

    crop_type = sys.argv[9].lower()
    growth_stage = sys.argv[10].lower()
    irrigation_type = sys.argv[11].lower()

    # Manual encoding (match with what your model was trained on)
    crop_dict = {"wheat": 0, "rice": 1, "maize": 2}
    growth_dict = {"early": 0, "mid": 1, "late": 2}
    irrigation_dict = {"drip": 0, "sprinkler": 1, "surface": 2}

    if crop_type not in crop_dict:
        print(f"Invalid crop_type '{crop_type}'. Allowed: {list(crop_dict.keys())}")
        sys.exit(1)
    if growth_stage not in growth_dict:
        print(f"Invalid growth_stage '{growth_stage}'. Allowed: {list(growth_dict.keys())}")
        sys.exit(1)
    if irrigation_type not in irrigation_dict:
        print(f"Invalid irrigation_type '{irrigation_type}'. Allowed: {list(irrigation_dict.keys())}")
        sys.exit(1)

    crop_encoded = crop_dict[crop_type]
    growth_encoded = growth_dict[growth_stage]
    irrigation_encoded = irrigation_dict[irrigation_type]

    features = [[
        mean_temp, max_temp, min_temp, humidity,
        wind_speed, pressure, radiation, day_of_year,
        crop_encoded, growth_encoded, irrigation_encoded
    ]]

    model = joblib.load("model/xgb_model.pkl")
    prediction = model.predict(features)[0]

    print(f"{prediction:.2f}")
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)

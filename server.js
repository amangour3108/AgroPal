const express = require("express");
const bodyParser = require("body-parser");
const { exec } = require("child_process");
const path = require("path");

const app = express();
app.set("view engine", "ejs");
app.use(express.static("public"));
app.use(bodyParser.urlencoded({ extended: true }));

app.get("/", (req, res) => {
  res.render("index");
});

app.post("/predict", (req, res) => {
  const {
    meanTemp,
    maxTemp,
    minTemp,
    humidity,
    windSpeed,
    pressure,
    radiation,
    dayOfYear,
    cropType,
    growthStage,
    irrigationType
  } = req.body;

  const pyCommand = `python scripts/predict_model.py ${meanTemp} ${maxTemp} ${minTemp} ${humidity} ${windSpeed} ${pressure} ${radiation} ${dayOfYear} ${cropType} ${growthStage} ${irrigationType}`;

  console.log("Executing command:", pyCommand);

  exec(pyCommand, (error, stdout, stderr) => {
    if (error) {
      console.error("Execution Error:", error.message);
      console.error("stderr:", stderr);
      return res.render("result", { prediction: `Error: ${stderr || error.message}` });
    }

    console.log("Prediction Output:", stdout);
    res.render("result", { prediction: stdout.trim() });
  });
});

app.get("/predict", (req,res) => {
  res.render("form.ejs");
})

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});

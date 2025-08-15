import { EMOTIONS_KEY, NO_MODEL } from "../Constants/emotionRecognizer.constant";
import * as tf from "@tensorflow/tfjs";
import { treatImg } from "./tensorflowImages";

const _predictTensor = (state, model, tfResizedImage) => {
  if (state.isModelSet) {
    const predict = Array.from(model.predict(tfResizedImage).dataSync());
    console.log("Raw model predictions:", predict);
    tfResizedImage.dispose();

    const dominantIndex = predict.indexOf(Math.max(...predict));
    const dominantEmotion = EMOTIONS_KEY[dominantIndex]; // returns "happy", "angry", etc.
    
    return dominantEmotion;
  } else {
    return NO_MODEL;
  }
};

const _predictImg = (emotionRecognizer, state, face) =>
  _predictTensor(state, emotionRecognizer, treatImg(face));

const predict = (emotionRecognizer, state, face) => {
  let prediction = "";
  tf.engine().startScope();
  tf.tidy(() => {
    prediction = _predictImg(emotionRecognizer, state, face);
  });
  tf.engine().endScope();
  return prediction;
};

export { predict };

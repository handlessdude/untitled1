const MODEL_PATH = 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/sqftToPropertyPrice/model.json';

let model = undefined;


import * as tf from '@tensorflow/tfjs';
console.log('Loaded TensorFlow.js - version: ' + tf.version.tfjs)

export async function loadModel() {

  model = await tf.loadLayersModel(MODEL_PATH);

  model.summary();

  

  // Create a batch of 1.

  const input = tf.tensor2d([[870]]);

  

  // Create a batch of 3

  const inputBatch = tf.tensor2d([[500], [1100], [970]]);


  // Actually make the predictions for each batch.

  const result = model.predict(input);

  const resultBatch = model.predict(inputBatch);

  

  // Print results to console.
  (result as tf.Tensor).print();  // Or use .arraySync() to get results back as array.

  (resultBatch as tf.Tensor).print(); // Or use .arraySync() to get results back as array.

//   For typescript, since predict() can return either a Tensor, or Tensor[], 
// depending on how many outputs your model has, you have to notify the compiler which one it is:
// (model.predict(...) as tf.Tensor).print()

  input.dispose();

  inputBatch.dispose();

  (result as tf.Tensor).dispose();

  (resultBatch as tf.Tensor).dispose();

  model.dispose();

}

//export default { loadModel }
// loadModel();
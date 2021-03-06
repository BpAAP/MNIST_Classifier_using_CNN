model_loaded = false;
is_drawing = false;
scale_factor = 8/12;

async function load_model(){
    const model = await tf.loadLayersModel('https://raw.githubusercontent.com/BpAAP/MNIST_Classifier_using_CNN/master/js_model/js_model/model.json');
    
    await model.save('localstorage://models/model');
    model_loaded = true;
    console.log("Model has successfully downloaded and saved.");
}

async function predict(array){
    if (!model_loaded){
        await load_model();
    }

    console.log("Making prediction...");
    const model = await tf.loadLayersModel('localstorage://models/model');
    in_tensor = tf.tensor([array]);    
    const prediction =  model.predict(in_tensor);   
    const values = prediction.argMax(-1).dataSync();
    const arr = Array.from(values);
    predict_txt.textContent = ("Looks like his is a/an: "+arr[0]).toString();
    console.log("Prediction made and displayed.");

}

const canvas = document.getElementById("imageCanvas");
const ctx = canvas.getContext("2d");

const predict_txt = document.getElementById("prediction_text");
const predict_btn = document.getElementById("predict_btn");
const reset_btn = document.getElementById("reset_btn");

function mouseDown(event){
    is_drawing=true;
    canvas_props = canvas.getBoundingClientRect()
    canvas_x = canvas_props['x'];
    canvas_y = canvas_props['y'];

    ctx.moveTo(event.pageX-canvas_x,event.pageY-canvas_y);
    ctx.beginPath();
}

function mouseMove(event){
    if (is_drawing){

        ctx.lineWidth = canvas.width / 28;
        ctx.lineCap = 'round';
        ctx.strokeStyle = 'black';
        canvas_props = canvas.getBoundingClientRect()
        canvas_x = canvas_props['x'];
        canvas_y = canvas_props['y'];

        ctx.lineTo(event.pageX-canvas_x,event.pageY-canvas_y);
        ctx.stroke();
    }
}

function mouseUp(event){
    is_drawing=false;
}

function resize(){
    const vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
    const vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);
    min_dim = Math.min(vw,vh)*scale_factor;
    canvas.width = min_dim;
    canvas.height = min_dim;
}

async function sample(e){
    predict_txt.textContent = "...Working...";
    console.log("Predict button pressed, sampling canvas...");

    cell_size = canvas.height / 28;
    
    array = [];
    for (i=0;i<28;i++){
        temparr = []
        for (j=0;j<28;j++){
            temparr.push(0.000000001);
        }
        array.push(temparr);
    }
    
    for (i=0;i<28;i++){
        for (j=0;j<28;j++){
            data  = ctx.getImageData(i*cell_size,j*cell_size,cell_size,cell_size).data;
            var sum = data.reduce(function(a, b){
                return a + b;
            }, 0);
            
            array[j][i] = sum;
            
        }
    }

    arr_max = 0;
    for(i=0;i<28;i++){
        for(j=0;j<28;j++){
            if(array[i][j]>arr_max){
                arr_max = array[i][j];
            }
        }
    }

    for(i=0;i<28;i++){
        for(j=0;j<28;j++){
            array[i][j] = (array[i][j] / arr_max);//*(-1)+1;
        }
    }
    
    await predict(array);
}

function reset(e){
    ctx.clearRect(0,0,canvas.width,canvas.height);
}

canvas.addEventListener('mousedown',mouseDown,false);
canvas.addEventListener('mousemove',mouseMove,false);
canvas.addEventListener('mouseup',mouseUp,false);

document.body.addEventListener('mouseup',mouseUp,false);

predict_btn.addEventListener('click',sample,false);
reset_btn.addEventListener('click',reset,false);

load_model();
resize();

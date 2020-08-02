

//Set up canvas
const canvas = document.getElementById("imageCanvas");
const ctx = canvas.getContext("2d");
ctx.strokeStyle = "black";

const prediction_text = document.getElementById("prediction_text");

scale_factor=1.2;

grid_matrix = [];

function reset_grid(){
    grid_matrix = []
    for (i=0;i<28;i++){
        templine = [];
        for (j=0;j<28;j++){
            templine.push(0.0);
        }
        grid_matrix.push(templine);
    }
    return grid_matrix
}

function setup_grid(){
    const vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0)
    const vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0)
    canvas_dim = Math.min(vw,vh)/scale_factor-(Math.min(vw,vh)/scale_factor)%28;

    canvas.width= canvas_dim;
    canvas.height = canvas_dim;

    

    for (i = 0;i<28;i++){
      ctx.beginPath();
      ctx.moveTo(i*canvas_dim/28,0);
      ctx.lineTo(i*canvas_dim/28,canvas_dim);
      ctx.stroke();
    }
    for (i = 0;i<28;i++){
        ctx.beginPath();
        ctx.moveTo(0,i*canvas_dim/28);
        ctx.lineTo(canvas_dim,i*canvas_dim/28);
        ctx.stroke();
      }
    
}

function render(grid_matrix){
    const vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0)
    const vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0)
    canvas_dim = Math.min(vw,vh)/scale_factor-(Math.min(vw,vh)/scale_factor)%28;

    pixel_size = canvas_dim/28;
    
    for(i=0;i<28;i++){
        for(j=0;j<28;j++){
            if(grid_matrix[j][i] === 1.0){
                ctx.fillStyle = "black";
                ctx.fillRect(i*pixel_size+1,j*pixel_size+1,pixel_size-2,pixel_size-2);
            }else{
                ctx.fillStyle = "white";
                ctx.fillRect(i*pixel_size+1,j*pixel_size+1,pixel_size-2,pixel_size-2);
            }
        }
    }
}


async function predict(grid_matrix){

    //Load model
   
    const model = await tf.loadLayersModel('https://powerful-tidy-sidewalk.glitch.me/js_model/js_model/model.json','https://powerful-tidy-sidewalk.glitch.me/js_model/js_model/group1-shard1of1.bin');
    
    in_tensor = tf.tensor([grid_matrix]);
    
    

    const prediction = model.predict(in_tensor);
    
    const values = prediction.argMax(-1).dataSync();
    const arr = Array.from(values);
        
    prediction_text.textContent = "I think this is a(n): "+arr[0].toString();
}

function changeSquare(event){
    const vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0)
    const vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0)
    canvas_dim = Math.min(vw,vh)/scale_factor-(Math.min(vw,vh)/scale_factor)%28;
    pixel_size = canvas_dim/28;
    i_pos = 0;
    j_pos = 0;
    
    for (i=0;i<28;i++){
        if(i*pixel_size<=event.pageY){
            if((i+1)*pixel_size>event.pageY){
                i_pos = i;
            }
        }
    }
    for (j=0;j<28;j++){
        if(j*pixel_size<=event.pageX){
            if((j+1)*pixel_size>event.pageX){
                j_pos = j;
            }
        }
    }
    if(grid_matrix[i_pos][j_pos]===1.0){
        grid_matrix[i_pos][j_pos] = 0.0;
    }else{
        grid_matrix[i_pos][j_pos] = 1.0;
    }
    render(grid_matrix);
    predict(grid_matrix);
}

function resized(){
    setup_grid();
    render(grid_matrix);
}



setup_grid();
grid_matrix = reset_grid();

canvas.addEventListener("mousedown",changeSquare,false);
grid_matrix= [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0, 0, 0,
    0, 0, 0],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0.32941176, 0.7254902 , 0.62352941, 0.59215686,
    0.23529412, 0.14117647, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0.87058824, 0.99607843, 0.99607843, 0.99607843,
    0.99607843, 0.94509804, 0.77647059, 0.77647059, 0.77647059,
    0.77647059, 0.77647059, 0.77647059, 0.77647059, 0.77647059,
    0.66666667, 0.20392157, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0.2627451 , 0.44705882, 0.28235294, 0.44705882,
    0.63921569, 0.89019608, 0.99607843, 0.88235294, 0.99607843,
    0.99607843, 0.99607843, 0.98039216, 0.89803922, 0.99607843,
    0.99607843, 0.54901961, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0.06666667, 0.25882353, 0.05490196, 0.2627451 ,
    0.2627451 , 0.2627451 , 0.23137255, 0.08235294, 0.9254902 ,
    0.99607843, 0.41568627, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0.3254902 , 0.99215686,
    0.81960784, 0.07058824, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.08627451, 0.91372549, 1,
    0.3254902 , 0, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.50588235, 0.99607843, 0.93333333,
    0.17254902, 0, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0.23137255, 0.97647059, 0.99607843, 0.24313725,
    0, 0, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0.52156863, 0.99607843, 0.73333333, 0.01960784,
    0, 0, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0.03529412, 0.80392157, 0.97254902, 0.22745098, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0.49411765, 0.99607843, 0.71372549, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0.29411765,
    0.98431373, 0.94117647, 0.22352941, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0.0745098 , 0.86666667,
    0.99607843, 0.65098039, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.01176471, 0.79607843, 0.99607843,
    0.85882353, 0.1372549 , 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.14901961, 0.99607843, 0.99607843,
    0.30196078, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0.12156863, 0.87843137, 0.99607843, 0.45098039,
    0.00392157, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0.52156863, 0.99607843, 0.99607843, 0.20392157,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0.23921569, 0.94901961, 0.99607843, 0.99607843, 0.20392157,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0.4745098 , 0.99607843, 0.99607843, 0.85882353, 0.15686275,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0.4745098 , 0.99607843, 0.81176471, 0.07058824, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.        ],
   [0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0.        ]];

    for (i=0;i<28;i++){
        for(j=0;j<28;j++){
            if (grid_matrix[i][j]!== 0){
                grid_matrix[i][j] = 1;
            }
        }
    }
render(grid_matrix);
predict(grid_matrix);
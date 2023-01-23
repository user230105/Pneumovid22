
const fileTypes = [    
    "image/bmp",
    "image/gif",
    "image/jpeg",
    "image/pjpeg",
    "image/png"    
  ];

function makePrediction(files){
    console.log('change')
    if (files.length > 0) {        
        console.log('new image')
        //updating image on web browser
        const file = files[0];
        let validfile = updateImageDisplay(file);
        console.log(validfile)
        if (validfile){
            console.log('image to server')
            //uploading image to server
            uploadImagetoServer(file)
        }
    }
}

function updateImageDisplay(files) {
    
    if (files.length > 0) {        
        
        //updating image on web browser
        const file = files[0];
        
        if (validFileType(file)) {   
            //get image display element
            imgDisplay = document.getElementById("imageDisplay");
            imgDisplay.src = URL.createObjectURL(file);
            
            formItemImgName = document.getElementById("imgfilename");
            formItemImgName.value = file.name
            console.log(file.name)
        }
    }
    
}

function uploadImagetoServer(){
    console.log('uploading5')

    //get file type input
    imgFile = document.getElementById("myimage");

    //get result placeholder
    evalElement = document.getElementById("eval_result");
    evalElement.value = '...'
    console.log(imgFile.files)
    if(imgFile.files.length > 0){
        console.log('files')
        const image = imgFile.files[0];

         
        var data = new FormData()
        data.append('file', image);
        console.log('sending image to server')
        //set promise for response
        let resPromise = fetch('/prediction', {
            method: 'POST',
            body: data
        })

        //set promise for json content
        resPromise.then((res) => res.json()).then((prediction) => {
            console.log('Success:', prediction);
            evalElement.value = prediction.result
        }).catch((error) => {
            console.error('Error:', error);
        }); 
    }  
    
}

function validFileType(file) {
    return fileTypes.includes(file.type);
}
(() => {
  var mousePressed = false;
  var lastX, lastY;
  var canvas = $('#trimapCanvas');
  var ctx = canvas[0].getContext('2d');
  var clearButton = $('#clear-button');
  var toggleButton = $('#toggle-button');
  var postButton = $('#telescope-button');
  var srcImg = document.getElementById('sourceImage');
  
  const mouseDownHandler = function (e) {
    mousePressed = true;
    let x = e.pageX - $(this).offset().left;
    let y = e.pageY - $(this).offset().top;
    draw(x, y, false);
  };

  const mouseLeaveHandler = function () {
    mousePressed = false;
  }

  const mouseMoveHandler = function (e) {
    if (mousePressed) {
      let x = e.pageX - $(this).offset().left;
      let y = e.pageY - $(this).offset().top;
      draw(x, y, true);
    }
  };

  const mouseUpHandler = function () {
    mousePressed = false;
  };

  const draw = (x, y, isDown) => {
    if (isDown) {
      ctx.beginPath();
      ctx.strokeStyle = $('#colorSelect input:radio:checked').val();
      ctx.lineWidth = $('#lineWidth').val();
      ctx.lineJoin = 'round';
      ctx.moveTo(lastX, lastY);
      ctx.lineTo(x, y);
      ctx.closePath();
      ctx.stroke();
    }
    lastX = x; lastY = y;
  };

  const clearTrimap = function () {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  };

  const postToServer = function () {
    var trimapURL = canvas[0].toDataURL();
    var c = document.getElementById('imageCanvas');
    var ctx_2 = c.getContext('2d');
    var img = document.getElementById('sourceImage');
    ctx_2.drawImage(img, 0, 0);
    var imgURL = c.toDataURL();
    $.ajax({
      type: "POST",
      url: "/demo/",
      data: JSON.stringify({imgUrl: imgURL, trimapUrl: trimapURL }, null, '\t'),
      contentType: 'application/json;charset=UTF-8'
    }).done(showAlphamatte);
  };

  const showAlphamatte = function (data) {
    $('#alphamatteImage').attr('src', 'data:image/png;base64,' + data.result);
    $('#alphamatteImage').show();
    toggleButton.show();
    clearButton.hide();
    postButton.hide();
    canvas.hide();
  };

  const toggleAlphamatte = function() {
    $('#alphamatteImage').toggle();
  };

  canvas.on({
    mousedown: mouseDownHandler,
    mouseup: mouseUpHandler,
    mousemove: mouseMoveHandler,
    mouseleave: mouseLeaveHandler
  });

  clearButton.click(clearTrimap);

  postButton.click(postToServer);

  toggleButton.click(toggleAlphamatte);
})();

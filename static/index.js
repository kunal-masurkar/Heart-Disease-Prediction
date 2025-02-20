const predictBtn = document.querySelector("button")
const form = document.querySelector("form")


    const resultSpanEl = document.querySelector("span")
    const heartImg = document.querySelector("img")
    if(resultSpanEl.innerText == "Negative"){
        resultSpanEl.style.color = "Green"
        heartImg.style.filter = "hue-rotate(120deg)"
        heartImg.style.transition = "all 0.75s linear"
    }
    else
        resultSpanEl.style.color = "Red"

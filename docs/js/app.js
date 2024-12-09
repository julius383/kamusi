let searchIdx = null;

async function getIndex() {
  try {
    const response = await fetch("https://julius383.github.io/kamusi/static/search_index.json");
    if (!response.ok) {
      throw new Error(`Response status: ${response.status}`);
    }

    const json = await response.json();
    const idx = elasticlunr.Index.load(json);
    return idx;
  } catch (error) {
    console.error(error.message);
  }
}

function formatEntry(entry) {
  let output = "";
  output += `
    <span><strong>${entry.swahili}</strong></span>
    <span><em>\ ${entry.part_of_speech} \ </em></span>
    `;
  if (entry.ngeli !== null) {
    output += `<span><mark>${entry.ngeli}</mark></span>`
  }
  output += `<br />`
  defs = entry.english
    .map((e) => {
      return `<li>${e}</li>`;
    })
    .join("\n");
  output += `<ol>${defs}</ol> `;

  output += `</ol>`;
  if (entry.examples !== null) {
    exs = entry.examples
      .map((e) => {
        return `<span>${e[0]} - ${e[1]}</span><br />`;
      })
      .join("\n");
    output += `<blockquote>${exs}</blockquote>`;
  }
  return `<article>${output}</article>`;
}

function setResults(html) {
  let resEle = document.querySelector("#search-results");
  resEle.innerHTML = "<h3>Results</h3>\n" + html;
  if (resEle.style.display === "none") {
    resEle.style.display = "block";
  }
  let wordDay = document.querySelector("details");
  wordDay.open = false;
}

function setWordOfDay() {
  let wod = document.querySelector("#word-of-day");
  wod.innerHTML = formatEntry(searchIdx.documentStore.getDoc(2958));
}

function searchWord(query) {
  let results = searchIdx.search(query);
  let output = "";
  if (results.length > 0) {
    results.forEach((result) => {
      let entry = result.doc;
      // console.log(entry);
      output += formatEntry(entry);
    });
    setResults(output);
  } else {
    setResults(`<h3>No results found for query "${query}"`)
  }
}

function getRandomIntInclusive(min, max) {
  const minCeiled = Math.ceil(min);
  const maxFloored = Math.floor(max);
  return Math.floor(Math.random() * (maxFloored - minCeiled + 1) + minCeiled); // The maximum is inclusive and the minimum is inclusive
}

function randomWord() {
  let rndId = getRandomIntInclusive(1, 13179); // 13179 is the number of entries in our dictionary and the
  setResults(formatEntry(searchIdx.documentStore.getDoc(rndId)));
}

async function setup() {
  searchIdx = await getIndex();
  setWordOfDay();
  console.log("Setup done");
}

document.addEventListener("DOMContentLoaded", setup);
if ("serviceWorker" in navigator) {
  window.addEventListener("load", function () {
    navigator.serviceWorker
      .register("/serviceWorker.js")
      .then((_) => console.log("service worker registered"))
      .catch((err) => console.log("service worker not registered", err));
  });
}

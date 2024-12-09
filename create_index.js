import elasticlunr from "./website/js/elasticlunr"

const path = Bun.argv[2];
const file = Bun.file(path);

const contents = await file.json();

let idx = elasticlunr(function () {
  this.addField("swahili");
  this.addField("english");
  this.setRef("id");

});

for (let entry of contents) {
  idx.addDoc(entry);
}
const ifile = Bun.file("website/static/search_index.json");
await Bun.write(ifile, JSON.stringify(idx.toJSON()));
console.log('Created index at website/static/search_index.json')

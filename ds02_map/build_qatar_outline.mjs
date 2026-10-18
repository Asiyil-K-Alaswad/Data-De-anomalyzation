// Extract Qatar (and the neighbouring coastlines for context) from Natural Earth
// 1:10m (npm world-atlas@2.0.2), projected to km around Qatar's centre with the
// same local equirectangular projection the site map uses. Output: qatar_outline.json
// Usage: node build_qatar_outline.mjs <path/to/node_modules> <out.json>
import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";

const [nm, out] = process.argv.slice(2);
const require = createRequire(nm + "/");
const { feature } = require("topojson-client");
const d3 = await import(nm + "/d3-geo/src/index.js");

const R = 6371.0088, LAT0 = 25.3, LON0 = 51.2;
const rad = Math.PI / 180, cos0 = Math.cos(LAT0 * rad);
const topo = JSON.parse(readFileSync(nm + "/world-atlas/countries-10m.json"));
const countries = feature(topo, topo.objects.countries).features;
const wanted = ["Qatar", "Bahrain", "Saudi Arabia", "United Arab Emirates"];

// km east/north of (LAT0, LON0); d3 projections put y downward, so flip at the end.
// (d3 re-centres the raw projection on .center(), which supplies the offset.)
const proj = d3.geoProjection((l, p) => [l * cos0, p])
  .center([LON0, LAT0]).scale(R).translate([0, 0])
  .clipExtent([[-260, -260], [260, 260]]).precision(0.05);

const shapes = wanted.map(name => {
  const f = countries.find(c => c.properties.name === name);
  const rings = [];
  let ring;
  d3.geoStream(f, proj.stream({
    polygonStart() {}, polygonEnd() {},
    lineStart() { ring = []; }, lineEnd() { if (ring.length > 2) rings.push(ring); },
    point(x, y) { ring.push([+x.toFixed(2), +(-y).toFixed(2)]); },
    sphere() {},
  }));
  return { name, rings };
});
writeFileSync(out, JSON.stringify({ lat0: LAT0, lon0: LON0, source: "Natural Earth 1:10m via world-atlas@2.0.2", shapes }));
console.log(shapes.map(s => `${s.name}: ${s.rings.length} rings, ${s.rings.flat().length} pts`).join("\n"));

(function(){
  const AFRAME_SRC = 'https://aframe.io/releases/1.4.1/aframe.min.js';
  function init() {
    const scene = document.createElement('a-scene');
    document.body.appendChild(scene);
    const nodes = {};
    const edges = {};

    function layout(ns) {
      const pos = {};
      const n = Math.max(ns.length, 1);
      ns.forEach((node, i) => {
        const t = 2 * Math.PI * i / n;
        pos[node.id] = {x: Math.cos(t)*2, y: 1.6, z: Math.sin(t)*2};
      });
      return pos;
    }

    function render(data) {
      const npos = layout(data.nodes || []);
      (data.nodes || []).forEach(node => {
        let ent = nodes[node.id];
        if (!ent) {
          ent = document.createElement('a-sphere');
          ent.setAttribute('radius', 0.2);
          ent.setAttribute('color', '#1f77b4');
          const text = document.createElement('a-entity');
          text.setAttribute('text', `value: ${node.text || node.id}; align: center;`);
          ent.appendChild(text);
          scene.appendChild(ent);
          nodes[node.id] = ent;
        }
        const p = npos[node.id];
        ent.setAttribute('position', `${p.x} ${p.y} ${p.z}`);
      });
      const edgesData = (data.edges || []).map(e => Array.isArray(e) ? e : [e.source, e.target]);
      edgesData.forEach(pair => {
        const key = pair.join('-');
        let line = edges[key];
        const a = npos[pair[0]];
        const b = npos[pair[1]];
        if (!a || !b) return;
        if (!line) {
          line = document.createElement('a-entity');
          scene.appendChild(line);
          edges[key] = line;
        }
        line.setAttribute('line', `start: ${a.x} ${a.y} ${a.z}; end: ${b.x} ${b.y} ${b.z}; color: black`);
      });
    }

    const url = new URLSearchParams(location.search).get('ws') || 'ws://localhost:8766/ws';
    const ws = new WebSocket(url);
    ws.onmessage = (e) => {
      try { render(JSON.parse(e.data)); } catch(err) { console.error(err); }
    };
  }

  if (window.AFRAME) { init(); } else {
    const s = document.createElement('script');
    s.src = AFRAME_SRC;
    s.onload = init;
    document.head.appendChild(s);
  }
})();

import { Application, Container, Graphics, Text } from 'pixi.js';

const COLORS = {
  recommended: 0x355d4a,
  conditional: 0xd58b36,
  rejected: 0xa8503e,
};

export class WeaverScene {
  constructor() {
    this.app = null;
    this.world = null;
    this.destroyed = false;
    this.lastState = null;
    this.selectedId = null;
  }

  async init(host, { onNodeTap } = {}) {
    this.host = host;
    this.onNodeTap = onNodeTap;
    const app = new Application();
    this.app = app;
    await app.init({
      backgroundAlpha: 0,
      antialias: true,
      resizeTo: host,
      autoDensity: true,
      resolution: Math.min(window.devicePixelRatio || 1, 2),
    });
    if (this.destroyed || this.app !== app) {
      if (app.renderer) app.destroy(true, { children: true });
      return;
    }
    host.appendChild(app.canvas);
    this.world = new Container();
    app.stage.addChild(this.world);
    this.resizeObserver = new ResizeObserver(() => {
      if (this.lastState) this.render(this.lastState, this.selectedId);
    });
    this.resizeObserver.observe(host);
    if (this.lastState) this.render(this.lastState, this.selectedId);
  }

  render(state, selectedId = null) {
    this.lastState = state;
    this.selectedId = selectedId;
    if (!this.world || this.destroyed) return;
    this.world.removeChildren();
    const nodes = state?.nodes || [];
    if (!nodes.length) return;

    const width = Math.max(this.host.clientWidth, 320);
    const height = Math.max(this.host.clientHeight, 360);
    const paddingX = Math.min(100, width * 0.14);
    const paddingY = Math.min(110, height * 0.2);
    const columns = width < 620 ? 2 : nodes.length > 6 ? 3 : 2;
    const rows = Math.ceil(nodes.length / columns);
    const usableWidth = width - paddingX * 2;
    const usableHeight = height - paddingY * 2;
    const positioned = nodes.map((node, index) => {
      const column = index % columns;
      const row = Math.floor(index / columns);
      return {
        ...node,
        x: paddingX + (columns === 1 ? usableWidth / 2 : column * usableWidth / (columns - 1)),
        y: paddingY + (rows === 1 ? usableHeight / 2 : row * usableHeight / (rows - 1)),
      };
    });
    const byId = new Map(positioned.map((node) => [node.id, node]));

    for (const thread of state.threads || []) {
      const from = byId.get(thread.from);
      const to = byId.get(thread.to);
      if (!from || !to) continue;
      const line = new Graphics();
      const middleX = (from.x + to.x) / 2;
      const middleY = (from.y + to.y) / 2 - 24;
      line.moveTo(from.x, from.y);
      line.bezierCurveTo(from.x, middleY, to.x, middleY, to.x, to.y);
      line.stroke({ width: 1.5, color: 0x8e7655, alpha: 0.34 });
      this.world.addChild(line);
    }

    for (const node of positioned) {
      const selected = node.id === selectedId;
      const color = COLORS[node.status] || COLORS.conditional;
      const radius = Math.max(22, Math.min(38, 20 + node.score / 7));
      const graphic = new Graphics();
      graphic.circle(node.x, node.y, radius + (selected ? 9 : 5));
      graphic.stroke({ width: selected ? 3 : 1.5, color, alpha: selected ? 0.9 : 0.3 });
      graphic.circle(node.x, node.y, radius);
      graphic.fill({ color, alpha: selected ? 1 : 0.88 });
      graphic.eventMode = 'static';
      graphic.cursor = 'pointer';
      graphic.on('pointertap', () => this.onNodeTap?.(node.id));
      this.world.addChild(graphic);

      const score = new Text({
        text: String(node.score),
        style: {
          fontFamily: 'Arial, sans-serif',
          fontSize: 13,
          fontWeight: '700',
          fill: 0xfffbf2,
        },
      });
      score.anchor.set(0.5);
      score.position.set(node.x, node.y);
      score.eventMode = 'none';
      this.world.addChild(score);

      const shortName = node.id.split('/').pop() || node.id;
      const label = new Text({
        text: shortName.length > 22 ? `${shortName.slice(0, 20)}…` : shortName,
        style: {
          fontFamily: 'Arial, sans-serif',
          fontSize: 11,
          fontWeight: selected ? '700' : '500',
          fill: 0x292721,
          align: 'center',
        },
      });
      label.anchor.set(0.5, 0);
      label.position.set(node.x, node.y + radius + 10);
      label.eventMode = 'none';
      this.world.addChild(label);
    }
  }

  destroy() {
    this.destroyed = true;
    this.resizeObserver?.disconnect();
    const app = this.app;
    this.app = null;
    this.world = null;
    if (app?.renderer) app.destroy(true, { children: true });
  }
}

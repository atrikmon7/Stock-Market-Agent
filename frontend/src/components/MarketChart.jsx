import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export function MarketChart({ technicals }) {
  if (!technicals?.candles?.length) return null;

  const data = technicals.candles.map((candle) => ({
    ...candle,
    label: new Date(candle.time).toLocaleDateString("en-IN", {
      day: "2-digit",
      month: "short",
    }),
  }));

  return (
    <section className="chart-panel" aria-label="Price chart">
      <div className="chart-head">
        <div>
          <p>{technicals.yahoo_symbol}</p>
          <h2>{technicals.last_close}</h2>
        </div>
        <span className={technicals.change_pct >= 0 ? "gain" : "loss"}>
          {technicals.change_pct >= 0 ? "+" : ""}
          {technicals.change_pct}%
        </span>
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <AreaChart data={data} margin={{ left: 0, right: 6, top: 8, bottom: 0 }}>
          <defs>
            <linearGradient id="price" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10a37f" stopOpacity={0.32} />
              <stop offset="95%" stopColor="#10a37f" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#ececf1" vertical={false} />
          <XAxis dataKey="label" minTickGap={28} tickLine={false} axisLine={false} />
          <YAxis domain={["dataMin", "dataMax"]} tickLine={false} axisLine={false} width={48} />
          <Tooltip
            contentStyle={{
              border: "1px solid #dedee6",
              borderRadius: 8,
              boxShadow: "0 12px 30px rgba(15, 23, 42, 0.1)",
            }}
          />
          <Area type="monotone" dataKey="close" stroke="#10a37f" fill="url(#price)" strokeWidth={2} />
        </AreaChart>
      </ResponsiveContainer>
      <div className="metric-grid">
        <span>RSI {technicals.rsi_14 ?? "-"}</span>
        <span>EMA20 {technicals.ema_20}</span>
        <span>EMA50 {technicals.ema_50}</span>
        <span>ATR {technicals.atr_14 ?? "-"}</span>
      </div>
    </section>
  );
}

import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { HistogramBin } from "./api";

interface Props {
  data: HistogramBin[];
  homeTeam: string;
}

export function MarginHistogram({ data, homeTeam }: Props) {
  const chartData = data.map((b) => ({
    margin: `${b.bin_start.toFixed(0)}`,
    count: b.count,
    label: `${b.bin_start.toFixed(0)} to ${b.bin_end.toFixed(0)}`,
  }));

  return (
    <ResponsiveContainer width="100%" height={180}>
      <BarChart data={chartData}>
        <XAxis
          dataKey="margin"
          tick={{ fill: "#888", fontSize: 10 }}
          interval="preserveStartEnd"
        />
        <YAxis tick={{ fill: "#888", fontSize: 10 }} />
        <Tooltip
          contentStyle={{
            background: "#1a1a2e",
            border: "1px solid #333",
            borderRadius: 8,
          }}
          labelFormatter={(_, payload) =>
            payload?.[0]?.payload?.label
              ? `Margin (${homeTeam} +): ${payload[0].payload.label}`
              : ""
          }
        />
        <Bar dataKey="count" fill="#ff6b35" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

/** AFL team branding for broadcast-style UI. */

export interface TeamBrand {
  name: string;
  abbr: string;
  nickname: string;
  primary: string;
  secondary: string;
  gradient: string;
}

export const TEAMS: Record<string, TeamBrand> = {
  Adelaide: {
    name: "Adelaide",
    abbr: "ADE",
    nickname: "Crows",
    primary: "#E21937",
    secondary: "#002B5C",
    gradient: "linear-gradient(135deg, #E21937 0%, #002B5C 100%)",
  },
  "Brisbane Lions": {
    name: "Brisbane Lions",
    abbr: "BRI",
    nickname: "Lions",
    primary: "#A30046",
    secondary: "#FDBE57",
    gradient: "linear-gradient(135deg, #A30046 0%, #5c0028 100%)",
  },
  Carlton: {
    name: "Carlton",
    abbr: "CAR",
    nickname: "Blues",
    primary: "#031A29",
    secondary: "#FFFFFF",
    gradient: "linear-gradient(135deg, #031A29 0%, #0d3d5c 100%)",
  },
  Collingwood: {
    name: "Collingwood",
    abbr: "COL",
    nickname: "Magpies",
    primary: "#000000",
    secondary: "#FFFFFF",
    gradient: "linear-gradient(135deg, #1a1a1a 0%, #000000 100%)",
  },
  Essendon: {
    name: "Essendon",
    abbr: "ESS",
    nickname: "Bombers",
    primary: "#CC203C",
    secondary: "#000000",
    gradient: "linear-gradient(135deg, #CC203C 0%, #1a0508 100%)",
  },
  Fremantle: {
    name: "Fremantle",
    abbr: "FRE",
    nickname: "Dockers",
    primary: "#2A0D54",
    secondary: "#FFFFFF",
    gradient: "linear-gradient(135deg, #2A0D54 0%, #4a2080 100%)",
  },
  Geelong: {
    name: "Geelong",
    abbr: "GEE",
    nickname: "Cats",
    primary: "#003973",
    secondary: "#FFFFFF",
    gradient: "linear-gradient(135deg, #003973 0%, #0055a4 100%)",
  },
  "Gold Coast": {
    name: "Gold Coast",
    abbr: "GCS",
    nickname: "Suns",
    primary: "#DC0019",
    secondary: "#FFDD00",
    gradient: "linear-gradient(135deg, #DC0019 0%, #8a0010 100%)",
  },
  GWS: {
    name: "GWS",
    abbr: "GWS",
    nickname: "Giants",
    primary: "#F47920",
    secondary: "#4D4D4F",
    gradient: "linear-gradient(135deg, #F47920 0%, #c45a10 100%)",
  },
  Hawthorn: {
    name: "Hawthorn",
    abbr: "HAW",
    nickname: "Hawks",
    primary: "#4D2004",
    secondary: "#FBBF15",
    gradient: "linear-gradient(135deg, #4D2004 0%, #7a3508 100%)",
  },
  Melbourne: {
    name: "Melbourne",
    abbr: "MEL",
    nickname: "Demons",
    primary: "#CC203C",
    secondary: "#0F1131",
    gradient: "linear-gradient(135deg, #CC203C 0%, #0F1131 100%)",
  },
  "North Melbourne": {
    name: "North Melbourne",
    abbr: "NTH",
    nickname: "Kangaroos",
    primary: "#003E92",
    secondary: "#FFFFFF",
    gradient: "linear-gradient(135deg, #003E92 0%, #001a40 100%)",
  },
  "Port Adelaide": {
    name: "Port Adelaide",
    abbr: "POR",
    nickname: "Power",
    primary: "#0099CC",
    secondary: "#000000",
    gradient: "linear-gradient(135deg, #0099CC 0%, #004466 100%)",
  },
  Richmond: {
    name: "Richmond",
    abbr: "RIC",
    nickname: "Tigers",
    primary: "#FFD200",
    secondary: "#000000",
    gradient: "linear-gradient(135deg, #FFD200 0%, #b89600 100%)",
  },
  "St Kilda": {
    name: "St Kilda",
    abbr: "STK",
    nickname: "Saints",
    primary: "#ED1B2F",
    secondary: "#FFFFFF",
    gradient: "linear-gradient(135deg, #ED1B2F 0%, #8a0f1a 100%)",
  },
  Sydney: {
    name: "Sydney",
    abbr: "SYD",
    nickname: "Swans",
    primary: "#ED171F",
    secondary: "#FFFFFF",
    gradient: "linear-gradient(135deg, #ED171F 0%, #9a0f14 100%)",
  },
  "West Coast": {
    name: "West Coast",
    abbr: "WCE",
    nickname: "Eagles",
    primary: "#003E7E",
    secondary: "#F2A900",
    gradient: "linear-gradient(135deg, #003E7E 0%, #F2A900 100%)",
  },
  "Western Bulldogs": {
    name: "Western Bulldogs",
    abbr: "WBD",
    nickname: "Bulldogs",
    primary: "#014BA0",
    secondary: "#BD002B",
    gradient: "linear-gradient(135deg, #014BA0 0%, #BD002B 100%)",
  },
};

export function getTeam(name: string): TeamBrand {
  return (
    TEAMS[name] ?? {
      name,
      abbr: name.slice(0, 3).toUpperCase(),
      nickname: name,
      primary: "#444",
      secondary: "#888",
      gradient: "linear-gradient(135deg, #333 0%, #111 100%)",
    }
  );
}

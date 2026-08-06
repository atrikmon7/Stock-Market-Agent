const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export async function sendChat(payload) {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "The AI service could not answer right now.");
  }

  return response.json();
}

export async function searchCompanies(query) {
  const response = await fetch(
    `${API_BASE}/api/companies?q=${encodeURIComponent(query)}&limit=8`,
  );
  if (!response.ok) return [];
  return response.json();
}

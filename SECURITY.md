# Security Policy

## Supported version

The current beta line is `1.0.0-beta.1`. Beta releases may contain defects and are not validated against every Windows, driver, and model combination.

## Report a vulnerability

Do not publish credentials, private photos, model tokens, or a weaponizable vulnerability in a public issue. If private vulnerability reporting is enabled for the repository, use GitHub's **Security → Report a vulnerability** flow. Otherwise contact the project maintainer through a private channel before sharing technical details publicly.

Include the app version, Windows/Python/runtime versions, relevant model roles, concise reproduction steps, and a redacted log excerpt. Remove usernames, private file paths, tokens, and unrelated personal data.

## Security boundaries

The app validates common file headers, archive extraction paths, URL schemes, and local loopback endpoints. These checks reduce risk but are not a security proof. Treat model files and downloaded native runtimes as executable-adjacent untrusted inputs, keep GPU drivers current, and review diagnostic exports before sharing them.

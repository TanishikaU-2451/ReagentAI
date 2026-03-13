# =============================================================================
# ReagentAI Frontend Dockerfile
# =============================================================================
# Three-stage build for the Next.js frontend application.
#
# Stage 1 (deps):  Installs Node.js dependencies using the lockfile for
#                   reproducible builds.
# Stage 2 (build): Compiles the Next.js application in standalone mode,
#                   producing a self-contained output directory.
# Stage 3 (run):   Copies only the minimal standalone artefacts into a clean
#                   Alpine image and starts the Node.js server on port 3000.
#
# Build:
#   docker build -t reagentai-frontend frontend/nextjs_app
#
# Run:
#   docker run -p 3000:3000 reagentai-frontend
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 -- Install dependencies
# ---------------------------------------------------------------------------
FROM node:20-alpine AS deps

# libc6-compat is often required by native Node modules on Alpine.
RUN apk add --no-cache libc6-compat

WORKDIR /app

# Copy package manifest and lockfile first so that dependency installation is
# cached independently of source-code changes.
COPY package.json yarn.lock* package-lock.json* pnpm-lock.yaml* ./

# Install dependencies using whichever package manager lockfile is present.
# Falls back to plain npm install if no lockfile is detected.
RUN \
    if [ -f yarn.lock ]; then yarn --frozen-lockfile; \
    elif [ -f package-lock.json ]; then npm ci; \
    elif [ -f pnpm-lock.yaml ]; then corepack enable pnpm && pnpm i --frozen-lockfile; \
    else echo "No lockfile found -- running npm install" && npm install; \
    fi

# ---------------------------------------------------------------------------
# Stage 2 -- Build the Next.js application
# ---------------------------------------------------------------------------
FROM node:20-alpine AS build

WORKDIR /app

# Copy installed node_modules from the deps stage.
COPY --from=deps /app/node_modules ./node_modules

# Copy the rest of the application source code.
COPY . .

# Set the API URL at build time so that Next.js can inline it into the client
# bundle.  The default points at the backend service name used in
# docker-compose.yml; override with --build-arg if needed.
ARG NEXT_PUBLIC_API_URL=http://backend:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

# Build the Next.js app.  The output is written to .next/ and, when
# standalone mode is enabled in next.config.js, a self-contained server is
# produced at .next/standalone.
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 3 -- Production runtime
# ---------------------------------------------------------------------------
FROM node:20-alpine AS run

WORKDIR /app

# Run the application as a non-root user for security.
RUN addgroup --system --gid 1001 nodejs && \
    adduser  --system --uid 1001 nextjs

# Copy the standalone server output from the build stage.
# The standalone directory contains a minimal Node.js server and all the
# code needed to run the application without node_modules.
COPY --from=build /app/.next/standalone ./

# Copy static assets and the public folder -- these are not included in the
# standalone output by default.
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public

# Switch to the non-root user.
USER nextjs

# Expose the port that the Next.js server listens on.
EXPOSE 3000

# Tell Next.js to listen on all interfaces so Docker port mapping works.
ENV HOSTNAME="0.0.0.0" \
    PORT=3000

# Start the standalone Next.js server.
CMD ["node", "server.js"]

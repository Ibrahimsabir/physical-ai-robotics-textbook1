# Quick Start: Physical AI & Humanoid Robotics Book

## Prerequisites
- Node.js (version 18 or higher)
- npm or yarn package manager
- Git for version control
- Basic command line knowledge

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Navigate to the e-book directory**
   ```bash
   cd e-book
   ```

3. **Install dependencies**
   ```bash
   npm install
   # or
   yarn install
   ```

4. **Start the development server**
   ```bash
   npm start
   # or
   yarn start
   ```
   This will start a local development server at http://localhost:3000

5. **Open your browser** to http://localhost:3000 to view the book

## Adding New Content

1. **Create a new markdown file** in the appropriate module directory under `/docs/`
2. **Follow the naming convention**: Use descriptive, URL-friendly names with hyphens
3. **Add frontmatter** at the top of your markdown file:
   ```markdown
   ---
   title: Your Chapter Title
   sidebar_label: Short Label
   description: Brief description of the chapter
   keywords: [list, of, relevant, keywords]
   ---
   ```

4. **Update the sidebar** in `sidebars.js` to include your new content

## Building for Production

```bash
npm run build
# or
yarn build
```

This creates a static build in the `build/` directory that can be deployed to any static hosting service.

## Running Tests

```bash
npm test
# or
yarn test
```

## Development Workflow

1. Make changes to your markdown files
2. The development server automatically reloads with your changes
3. Test your changes in the browser
4. Commit your changes with descriptive commit messages
5. Push changes to the repository

## Key Directories

- `/docs/` - Main book content organized by modules
- `/src/components/` - Custom React components for interactive elements
- `/static/` - Static assets like images and code examples
- `/blog/` - Optional blog posts related to the content

## Common Tasks

- **Add a new chapter**: Create a new `.md` file in the appropriate module directory
- **Add a new module**: Create a new directory in `/docs/` and update `sidebars.js`
- **Add custom components**: Create new React components in `/src/components/`
- **Add images**: Place images in `/static/img/` and reference with absolute paths
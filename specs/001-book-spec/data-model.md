# Data Model: Physical AI & Humanoid Robotics Book

## Content Document
- **id**: Unique identifier for the document
- **title**: Display title of the chapter/section
- **module**: Module identifier (e.g., "module-1-robotic-nervous-system")
- **slug**: URL-friendly identifier
- **authors**: Array of author names
- **tags**: Array of topic tags for search and filtering
- **prerequisites**: Array of prerequisite concepts/chapters
- **learningObjectives**: Array of learning objectives for the section
- **duration**: Estimated reading/learning time in minutes
- **difficulty**: Level (beginner, intermediate, advanced)
- **content**: Markdown content with embedded components
- **metadata**: Additional metadata like date created, last updated, etc.

## Module
- **id**: Module identifier (e.g., "module-1-robotic-nervous-system")
- **title**: Module title
- **description**: Brief description of the module
- **order**: Sequential order of the module in the book
- **learningOutcomes**: Array of learning outcomes for the module
- **chapters**: Array of chapter IDs that belong to this module
- **prerequisites**: Array of prerequisite modules or concepts

## Exercise
- **id**: Unique exercise identifier
- **title**: Exercise title
- **moduleId**: Module this exercise belongs to
- **chapterId**: Chapter this exercise belongs to
- **type**: Type of exercise (quiz, coding, simulation, etc.)
- **difficulty**: Level (beginner, intermediate, advanced)
- **instructions**: Detailed instructions for the exercise
- **solution**: Solution or reference implementation
- **resources**: Array of additional resources needed for the exercise

## CodeExample
- **id**: Unique identifier for the code example
- **title**: Title of the code example
- **moduleId**: Module this example belongs to
- **chapterId**: Chapter this example belongs to
- **language**: Programming language (Python, C++, C#, etc.)
- **code**: Source code content
- **description**: Explanation of what the code does
- **requirements**: System requirements to run the code
- **output**: Expected output or behavior

## MediaAsset
- **id**: Unique identifier for the media asset
- **filename**: Original filename
- **path**: Relative path from static directory
- **type**: Asset type (image, video, 3d-model, etc.)
- **altText**: Alternative text for accessibility
- **caption**: Descriptive caption
- **usageRights**: Information about usage rights
- **moduleId**: Module where asset is used (optional)
- **chapterId**: Chapter where asset is used (optional)

## NavigationItem
- **id**: Unique identifier for navigation item
- **label**: Display label for navigation
- **href**: URL or path to the content
- **parentId**: Parent navigation item ID (for nested structure)
- **order**: Display order within parent
- **type**: Type of item (link, category, etc.)
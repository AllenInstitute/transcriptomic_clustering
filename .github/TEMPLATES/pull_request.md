<!--The following template outlines the expectations for a PR created
by a member of Marmot team. All of the following sections should be filled
out for a PR to be reviewed quickly and correctly. Just remember, don't take yourself
too too seriously ;)-->

## Overview:
<!--Give a brief overview of the issue you are solving. Succinctly
explain the GitHub issue you are addressing and the underlying problem
of the ticket. The commit header and body should also include this
message, for good commit messages see https://chris.beams.io/posts/git-commit/

Example: This issue is targetting adding a new feature to do <task>. This task is
desired for <use case of task> and produces <new outputs>. I added the feature to
<location of added feature>, it takes as input <list of inputs> and outputs <list of
outputs>.-->

## Addresses:
<!--Add a link to the issue this is resolving, either url or #<issue_number>

Example: Addresses #<issue number>-->

## Solution:
<!--Outline your solution to the previously described issue and
underlying cause. This section should include a brief description of
your proposed solution and how it addresses the cause of the ticket.
Don't kill yourself with detail, overviews are fine no need to go line by 
line.

Example: This was a complex feature to add as it required generating <output>
from <input> this required the use of additional inputs <other inputs>. This feature 
generates this output by doing the following steps <numbered list of steps with descriptions>-->

## Changes:
<!--Include a bulleted list or check box list of the implemented changes
in brief, as well as the addition of supplementary materials(unit tests,
integration tests, etc

Example: Below is a bulleted list of changes that were made to the codebase
<Bulleted list of changes>-->

## Validation:
<!--Describe how you have validated that your solution addresses the
root cause of the ticket. What have you done to ensure that your
addition is bug free and works as expected? Please provide specific
instructions so we can reproduce and list any relevant details about
your configuration
Example: I have validated my PR by doing the following, <description of how
and what data was used to validate>

### Screenshots: <Screenshots showing validation>
### Script: <script to reproduce the validation (can be used on any branch to validate it fixes the problem and the problem exists)>-->
### Screenshots:
### Script to reproduce error and fix:

## Checklist
If the answer is not yes to any item, please explain why you still want to have this pull request
- [ ] My code follows
      [Allen Institute Contribution Guidelines](https://github.com/AllenInstitute/AllenSDK/blob/master/CONTRIBUTING.md)
- [ ] I have performed a self review of my own code
- [ ] My code is well-documented, and the docstrings conform to
      [Numpy Standards](https://numpydoc.readthedocs.io/en/latest/format.html)
- [ ] I have run PyLint on my code (if CI is not set up to run this)
- [ ] I have updated the documentation of the repository where
      appropriate
- [ ] I have added unittests as necessary
- [ ] I have updated the CHANGELOG.md to include the new features/bug fixes/changes included in the new release.

## Potential Impact:
<!--Use this section to explain what’s (the worst) impact if this pull request does not work as expected? Ask yourself
could this break a production service? If so, what have you done to mitigate the potential impact?

Example: This change could potentially impact the results of <insert functions or classes impacted> because they 
now make use of this new feature for generating their outputs of <list outputs>. I have validated that with the current
testing data these outputs do not change, but their may be a bug that was not caught.-->

## Notes:
<!-- Use this section to add anything you think worth mentioning to the
reader of the issue.

Example: This new feature provides a lot of infrastructure for providing <new feature or service> down the road. We might want to consider
this down the road. I also noticed that we could improve <other feature> by replacing some of our logic with <new logic or method that could be introduced>.-->

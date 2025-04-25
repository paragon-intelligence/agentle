# type: ignore
# /// script
# dependencies = [
#   "rich",
#   "questionary",
#   "msgspec",
#   "toml",
#   "tomlkit",
#   "pyfiglet",
# ]
# ///

import os
import re
import shutil
import subprocess
import sys
import signal
from enum import Enum, auto
from typing import List, Optional

import msgspec
import questionary
import toml
import tomlkit
from questionary.prompts.common import Choice
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.theme import Theme
from rich.traceback import install

install()  # Better traceback formatting with Rich

# Force Python's default SIGINT handler so Ctrl+C isn't swallowed
signal.signal(signal.SIGINT, signal.SIG_DFL)


def parse_version(version: str) -> list[int]:
    """Parse a string version like 'v0.1.2' into a list of ints [0,1,2]."""
    return list(map(int, version.lstrip("v").split(".")))


def compare_releases(version1: str, version2: str) -> int:
    """Compare two version strings ('v0.1.2'). Return -1, 0, or 1."""
    parsed_version1 = parse_version(version1)
    parsed_version2 = parse_version(version2)
    if parsed_version1 < parsed_version2:
        return -1
    elif parsed_version1 > parsed_version2:
        return 1
    else:
        return 0


class ReleaseType(Enum):
    MAJOR = auto()
    FEATURE = auto()
    PATCH = auto()


def clear_console() -> None:
    """Clear the console screen."""
    os.system("cls" if os.name == "nt" else "clear")


def display_ascii_art():
    """Display an ASCII art title with gradient."""
    from rich.text import Text
    from pyfiglet import Figlet

    f = Figlet(font="big")
    ascii_art = f.renderText("Agentle")
    console.print(Text(ascii_art, style="bold magenta"), justify="center")


# Define console with custom theme
custom_theme = Theme(
    {
        "success": "bold green",
        "error": "bold red",
        "question": "bold cyan",
        "info": "cyan",
        "warning": "bold yellow",
    }
)
console = Console(theme=custom_theme)


def run_command(
    cmd: List[str],
    description: str,
    acceptable_exit_codes: Optional[List[int]] = None,
    success_output_patterns: Optional[List[str]] = None,
):
    """Run a system command with progress indication and error handling."""
    acceptable_exit_codes = acceptable_exit_codes or [0]
    success_output_patterns = success_output_patterns or []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task(description)
        result = subprocess.run(cmd, capture_output=True, text=True)
        stdout_lower = result.stdout.lower()
        stderr_lower = result.stderr.lower()
        combined_output = stdout_lower + stderr_lower

        if result.returncode not in acceptable_exit_codes:
            # Check if any success pattern matches the output
            if any(
                re.search(pattern, combined_output)
                for pattern in success_output_patterns
            ):
                progress.update(task, completed=100)
                console.print(
                    f"[success]{description} completed successfully (handled non-zero exit code).[/success]"
                )
            else:
                progress.update(task, completed=100)
                console.print(Panel(f"Command {' '.join(cmd)} failed.", style="error"))
                console.print(
                    Panel(
                        f"[bold]STDOUT:[/bold]\n{result.stdout}\n[bold]STDERR:[/bold]\n{result.stderr}",
                        style="error",
                    )
                )
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, output=result.stdout, stderr=result.stderr
                )
        else:
            progress.update(task, completed=100)
            console.print(f"[success]{description} completed successfully.[/success]")


def main():
    clear_console()
    display_ascii_art()

    # Loading animation while checking environment
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Checking environment...", total=None)
        # Ensure GitHub CLI is installed
        gh_cli = shutil.which("gh")
        if not gh_cli:
            progress.stop()
            console.print(
                "[error]GitHub CLI ('gh') is required to run this script.[/error]"
            )
            raise ValueError("GitHub CLI ('gh') is required to run this script.")

        # Fetch latest release tag
        try:
            result = subprocess.run(
                [gh_cli, "release", "list", "--json", "tagName"],
                env={**os.environ, "GH_PAGER": "cat"},
                capture_output=True,
                text=True,
                check=True,
            )
            progress.update(
                task, description="Environment check completed.", completed=100
            )
        except subprocess.CalledProcessError as e:
            progress.stop()
            console.print("[error]Failed to fetch releases from GitHub.[/error]")
            raise e

    releases = msgspec.json.decode(result.stdout)

    # If no release found, ask user if they want to start from v0.0.1 or specify a custom version
    if not releases:
        console.print("[warning]No existing releases found on GitHub.[/warning]")
        version_choice = questionary.select(
            "No releases found. Do you want to use v0.0.1 or set your own version?",
            choices=["Use v0.0.1", "Specify my own version"],
        ).unsafe_ask()

        if version_choice is None:
            console.print("[warning]Prompt was cancelled. Exiting...[/warning]")
            sys.exit(130)

        if version_choice == "Use v0.0.1":
            tag_version = "v0.0.1"
        else:
            custom_version = questionary.text(
                "Enter your custom version (e.g., v0.1.0):"
            ).unsafe_ask()
            if custom_version is None:
                console.print("[warning]Prompt was cancelled. Exiting...[/warning]")
                sys.exit(130)
            if not custom_version.startswith("v"):
                custom_version = f"v{custom_version}"
            tag_version = custom_version
    else:
        tag_version = releases[0].get("tagName") if releases else "v0.0.0"

    # Release type selection with beautiful prompts
    console.print(
        Panel(
            Markdown("### What kind of release are you doing?"),
            title="Choose Release Type",
            title_align="left",
            border_style="blue",
        )
    )
    choices = [
        Choice(title="🐛 Patch", value=ReleaseType.PATCH),
        Choice(title="✨ Feature", value=ReleaseType.FEATURE),
        Choice(title="🚀 Major", value=ReleaseType.MAJOR),
    ]
    user_response = questionary.select(
        "",
        choices=choices,
        style=questionary.Style(
            [
                ("question", "fg:#00FF00 bold"),
                ("answer", "fg:#FFA500 bold"),
                ("pointer", "fg:#FF0000 bold"),
            ]
        ),
    ).unsafe_ask()

    if user_response is None:
        console.print("[warning]Prompt was cancelled. Exiting...[/warning]")
        sys.exit(130)

    # Double confirmation for major or feature releases
    if user_response in [ReleaseType.MAJOR, ReleaseType.FEATURE]:
        for _ in range(2):
            confirm_result = questionary.confirm(
                f"Are you sure you want to proceed with a {user_response.name} release?",
                default=False,
                style=questionary.Style([("question", "fg:#FF69B4 bold")]),
            ).unsafe_ask()
            if confirm_result is None:
                console.print("[warning]Prompt was cancelled. Exiting...[/warning]")
                sys.exit(130)

            if not confirm_result:
                console.print(
                    Panel(
                        "[warning]Release process terminated.[/warning]",
                        style="warning",
                    )
                )
                return

    # Load and update pyproject.toml
    pyproject = toml.load("pyproject.toml")
    project = pyproject.get("project", None)
    if not project or "version" not in project:
        console.print(
            Panel(
                "[error]The 'version' field is required in 'pyproject.toml'.[/error]",
                style="error",
            )
        )
        raise ValueError("The 'version' field is required in 'pyproject.toml'.")

    # Calculate new version
    parsed_version = parse_version(tag_version)
    if user_response == ReleaseType.MAJOR:
        parsed_version[0] += 1
        parsed_version[1] = 0
        parsed_version[2] = 0
    elif user_response == ReleaseType.FEATURE:
        parsed_version[1] += 1
        parsed_version[2] = 0
    else:
        parsed_version[2] += 1

    new_version = f"v{'.'.join(map(str, parsed_version))}"
    console.print(f"[info]New version calculated: {new_version}[/info]")

    # Git operations
    confirm_git_ops = questionary.confirm(
        "Add, commit, and push all changes before releasing?", default=True
    ).unsafe_ask()
    if confirm_git_ops is None:
        console.print("[warning]Prompt was cancelled. Exiting...[/warning]")
        sys.exit(130)

    if confirm_git_ops:
        commands = [
            (["git", "add", "."], "Staging changes"),
            (
                ["git", "commit", "-m", new_version],
                f"Committing changes with message '{new_version}'",
            ),
            (["git", "push"], "Pushing changes to repository"),
        ]
        for cmd, description in commands:
            # For 'git commit', handle the 'nothing to commit' case
            if cmd[1] == "commit":
                run_command(
                    cmd,
                    description,
                    acceptable_exit_codes=[0, 1],
                    success_output_patterns=[r"nothing to commit"],
                )
            else:
                run_command(cmd, description)

    # Update pyproject.toml
    pyproject["project"]["version"] = new_version
    with open("pyproject.toml", "w") as f:
        f.write(tomlkit.dumps(pyproject))
    console.print(
        f"[success]Updated 'pyproject.toml' with new version {new_version}.[/success]"
    )

    # Ensure CHANGELOG.md exists
    changelog_path = "CHANGELOG.md"
    if not os.path.exists(changelog_path):
        with open(changelog_path, "w") as f:
            f.write("# Changelog\n\n")
        console.print(Panel(f"{changelog_path} created.", style="info"))
    else:
        # Check if the current version in CHANGELOG.md matches the new version
        with open(changelog_path, "r") as f:
            content = f.read()

        # Regex pattern to match version strings like v0.1.2 or v0.2.1
        match = re.search(r"#* v?(\d+\.\d+\.\d+)", content)

        if match:
            current_version = match.group(1).strip()
        else:
            current_version = "v0.0.0"  # Default version if no version header is found

        if current_version and new_version:
            if compare_releases(current_version, new_version) != 0:
                confirm_changelog_overwrite = questionary.confirm(
                    f"The version in {changelog_path} ({current_version}) does not match the new version ({new_version}). Do you want to overwrite the file?",
                    default=True,
                ).unsafe_ask()
                if confirm_changelog_overwrite is None:
                    console.print("[warning]Prompt was cancelled. Exiting...[/warning]")
                    sys.exit(130)

                if confirm_changelog_overwrite:
                    with open(changelog_path, "w") as f:
                        f.write(f"# Changelog\n\n## {new_version}\n\n- \n")
                    console.print(
                        f"[info]Overwritten {changelog_path} with new version header.[/info]"
                    )
                else:
                    console.print("[info]Keeping existing CHANGELOG.md content.[/info]")
        else:
            console.print(
                f"[warning]Could not determine version from {changelog_path}. Overwriting with new version.[/warning]"
            )
            with open(changelog_path, "w") as f:
                f.write(f"# Changelog\n\n## {new_version}\n\n- \n")

    # Git commit and push for changes in pyproject.toml and CHANGELOG.md
    commands = [
        (["git", "add", "pyproject.toml", changelog_path], "Staging version changes"),
        (
            ["git", "commit", "-m", f"Release {new_version}"],
            f"Committing version changes with message 'Release {new_version}'",
        ),
        (["git", "push"], "Pushing version changes to repository"),
    ]
    for cmd, description in commands:
        # Handle 'git commit' with potential 'nothing to commit' message
        if cmd[1] == "commit":
            run_command(
                cmd,
                description,
                acceptable_exit_codes=[0, 1],
                success_output_patterns=[r"nothing to commit"],
            )
        else:
            run_command(cmd, description)

    # Create release using GitHub CLI
    release_title = questionary.text(
        "Enter the release title (this will appear as the heading of the release on GitHub). Leave blank to use the default version tag:",
        default=new_version,
    ).unsafe_ask()
    if release_title is None:
        console.print("[warning]Prompt was cancelled. Exiting...[/warning]")
        sys.exit(130)

    extra_flags = questionary.text(
        "Enter additional GitHub CLI flags (optional):"
    ).unsafe_ask()
    if extra_flags is None:
        console.print("[warning]Prompt was cancelled. Exiting...[/warning]")
        sys.exit(130)

    try:
        gh_release_cmd = [
            gh_cli,
            "release",
            "create",
            new_version,
            "-F",
            changelog_path,
            "-t",
            release_title or new_version,
        ] + (extra_flags.split() if extra_flags else [])

        run_command(gh_release_cmd, f"Creating GitHub release {new_version}")
        console.print(
            Panel(
                f"Release [success]{new_version}[/success] created successfully!",
                style="success",
            )
        )
    except subprocess.CalledProcessError as e:
        console.print(Panel(f"Failed to create release: {e}", style="error"))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("[warning]Received Ctrl+C! Exiting immediately...[/warning]")
        sys.exit(130)

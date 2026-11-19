import type { AnalysisResult } from '../../utils/repoAnalyzer'
import { CopyButton, Panel } from './primitives'
import styles from './analyzer.module.css'

export function BadgePanel({ grade }: Readonly<{ grade: AnalysisResult['grade'] }>) {
  const encodedGrade = encodeURIComponent(grade.letter)
  const encodedColor = grade.color.replace('#', '')
  const badgeUrl = `https://img.shields.io/badge/claude_cost_optimizer-${encodedGrade}-${encodedColor}?style=for-the-badge`
  const markdown = `[![TokenTariff Grade](${badgeUrl})](https://atluri0001.github.io/token-tariff/analyzer)`
  return (
    <Panel title="Your Badge">
      <div className={styles.badgePreview}>
        <img src={badgeUrl} alt={`Grade: ${grade.letter}`} />
      </div>
      <code className={styles.badgeCode}>{markdown}</code>
      <CopyButton text={markdown} label="Copy Badge Markdown" doneLabel="Copied!" />
    </Panel>
  )
}

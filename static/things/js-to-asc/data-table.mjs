const html = String.raw;

function parseCSV(data) {
  return data
    .split("\n")
    .filter(v => v.length >= 1)
    .map(v => v.split(",").map(v => v.trim()));
}

export class DataTable {
    constructor(header, rows) {
        this.header = header;
        this.rows = rows;
    }

    static fromCSV(csv) {
        const rawData = parseCSV(csv);
        return new DataTable(rawData[0].map(name => ({name, classList: []})), rawData.slice(1));
    }

    rowMatchesPredicate(row, predicate) {
        for(let [prop, vals] of Object.entries(predicate)) {
            if(typeof vals === "string") {
                vals = [vals];
            }
            const propIdx = this.header.findIndex(head => head.name.toLowerCase() === prop.toLowerCase());
            if(!vals.includes(row[propIdx])) {
                return false
            }
        }
        return true
    }

    filter(...predicates) {
        const data = this.rows.filter(row => predicates.some(predicate => this.rowMatchesPredicate(row, predicate)));
        return new DataTable(this.header.slice(), data);
    }

    toHTML() {
        return html`
            <table class="data-table">
                <thead>
                    <tr>
                        ${this.header.map(head => html`<th class="${head.classList.join(" ")}">${head.name}</th>`).join("")}
                    </tr>
                </thead>
                <tbody>
                    ${
                        this.rows.map(row => html`
                            <tr>
                                ${row.map((col, i) => html`<td class="${this.header[i].classList.join(" ")}">${col}</td>`).join("")}
                            </tr>
                        `).join("")
                    }
                </tbody>
            </table>
        `;
    }
}
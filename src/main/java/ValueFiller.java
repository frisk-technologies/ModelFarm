import org.apache.commons.csv.CSVRecord;

public class ValueFiller {
    private String[] targetAttributes;
    private String[] customMissingValues;
    private FillType fillType;

    public ValueFiller(String[] targetAttributes, FillType fillType, String... customMissingValues) {
        this.targetAttributes = targetAttributes;
        this.fillType = fillType;
        this.customMissingValues = customMissingValues;
    }

    public CSVRecord fillMissingValues(CSVRecord record, String target) {
        for (String targetAttribute : targetAttributes) {
            if (!record.isMapped(target))
                throw new RuntimeException("Target attribute non-existent");
            String value = record.get(targetAttribute);
            if (value == null || value.isEmpty()) {
                switch (fillType) {
                    case NUMERIC:
                        record = fillNumeric(record, target);
                        break;
                    case NOMINAL:
                        record = fillNominal(record, target);
                        break;
                    default:
                        throw new IllegalStateException("Unexpected value: " + fillType);
                }
            }
        }
        return record;
    }

    private CSVRecord fillNominal(CSVRecord record, String target) {
        return record;
    }

    private CSVRecord fillNumeric(CSVRecord record, String target) {
        return record;
    }

    public enum FillType {
        NUMERIC,
        NOMINAL
    }
}

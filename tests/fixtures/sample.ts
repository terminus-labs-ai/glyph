/** A sample class with docs */
export class MyClass extends BaseClass implements Serializable {
    private name: string;

    /** Constructor docs */
    constructor(name: string) {
        this.name = name;
    }

    /** Gets the name */
    public getName(): string {
        return this.name;
    }

    static create(): MyClass {
        return new MyClass("default");
    }

    async fetchData(): Promise<Data> {
        return await fetch("/data");
    }
}

/** A standalone function */
export function greet(name: string): string {
    return `Hello, ${name}`;
}

/** Arrow function */
export const add = (a: number, b: number): number => a + b;

const internal = (x: number) => x * 2;

/** MyInterface docs */
export interface MyInterface {
    id: number;
    getName(): string;
}

/** Status enum */
export enum Status {
    Active = "active",
    Inactive = "inactive",
}

/** Type alias */
export type Result<T> = { ok: true; value: T } | { ok: false; error: Error };

/**
 * Documented function
 * @param name - The name
 * @returns A greeting
 * @deprecated Use greetV2 instead
 */
function documented(name: string): string {
    return name;
}

abstract class AbstractBase {
    abstract doWork(): void;

    protected helper(): void {}
}

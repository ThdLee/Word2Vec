package util;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map.Entry;

public class MapCount<T> {
    private LinkedHashMap<T, Integer> hm = null;
    private ArrayList<T> indexMap = null;

    private int size = 0;

    public MapCount() {
        this.hm = new LinkedHashMap<>();
        this.indexMap = new ArrayList<>();
    }

    public MapCount(int initialCapacity) {
        this.hm = new LinkedHashMap<>(initialCapacity);
        this.indexMap = new ArrayList<>(initialCapacity);
    }

    public void add(T t, int n) {
        if (!hm.containsKey(t)) indexMap.add(t);
        Integer integer = null;
        if ((integer = (Integer)this.hm.get(t)) != null) {
            this.hm.put(t, Integer.valueOf(integer.intValue() + n));
        } else {
            this.hm.put(t, Integer.valueOf(n));
        }
        size += n;
    }

    public void add(T t) {
        this.add(t, 1);
    }

    public int size() {
        return size;
    }

    public void remove(T t) {
        this.hm.remove(t);
    }

    public LinkedHashMap<T, Integer> get() {
        return this.hm;
    }

    public T getKey(int index) {
        if (index < 0 || index >= indexMap.size()) return null;
        return indexMap.get(index);
    }

    public String getDic() {
        Iterator iterator = this.hm.entrySet().iterator();
        StringBuilder sb = new StringBuilder();
        Entry next = null;

        while (iterator.hasNext()) {
            next = (Entry)iterator.next();
            sb.append(next.getKey()).append("\t").append(next.getValue()).append("\n");
        }

        return sb.toString();
    }

}

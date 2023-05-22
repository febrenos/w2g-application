import React, { useState } from "react";
import './style.css'

function MenuItem({ item, isActive, onClick }) {
  const handleClick = () => {
    onClick(item);
  };

  return (
    <li onClick={handleClick} className={isActive ? "active" : ""}>
      {item}
    </li>
  );
}

function Menu({ items, onItemClick }) {
  const [activeItem, setActiveItem] = useState(null);

  const handleClick = (item) => {
    setActiveItem(item);
    onItemClick(item);
  };

  return (
    <ul>
      {items.map((item, index) => (
        <MenuItem
          key={index}
          item={item}
          isActive={activeItem === item}
          onClick={handleClick}
        />
      ))}
    </ul>
  );
}

function App() {
  const [activeItem, setActiveItem] = useState(null);

  const handleItemClick = (item) => {
    setActiveItem(item);
  };

  return (
    <div>
      <Menu
        items={["Item 1", "Item 2", "Item 3"]}
        onItemClick={handleItemClick}
      />
      {activeItem && <div>{activeItem} is active</div>}
    </div>
  );
}

export default App;